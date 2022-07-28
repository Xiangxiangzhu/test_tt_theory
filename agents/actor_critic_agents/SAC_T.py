import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import copy
from agents.Base_Agent import Base_Agent
from utilities.OU_Noise import OU_Noise
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from models import EncoderLayer
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class TSAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github
    implementation https://github.com/pranz24/pytorch-soft-actor-critic.
    It is an actor-critic algorithm where the agent is also trained
    to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", \
            "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        assert self.config.hyperparameters["Actor"][
                   "final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
        self.config.hyperparameters["Actor"]["final_layer_activation"] = None
        assert self.config.hyperparameters["Actor"][
                   "final_layer_activation"] is None, "Final actor layer must be None in this setting"
        self.hyperparameters = config.hyperparameters

        self.n_feature = self.state_size
        self.n_history = 1
        self.input_dim = 1 * self.n_history

        self.state_dim = 128
        self.nheads = 8
        self.node_num = self.n_feature
        self.dropout = 0.
        self.emb = EncoderLayer(self.input_dim, self.state_dim, self.nheads, self.node_num, self.dropout)
        # self.temb = EncoderLayer(self.input_dim, self.state_dim, self.nheads, self.node_num, self.dropout)
        self.critic_local = self.create_NN(input_dim=self.n_feature + self.action_size, output_dim=1,
                                           key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.n_feature + self.action_size, output_dim=1,
                                             key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.n_feature + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.n_feature + self.action_size, output_dim=1,
                                              key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)
        self.actor_local = self.create_NN(input_dim=self.n_feature, output_dim=self.action_size * 2,
                                          key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

        self.stacked_obs = []
        self.adj = self.adj = torch.FloatTensor(np.ones((self.state_size, self.state_size))).unsqueeze(0)
        self.emb_optimizer = torch.optim.Adam(self.emb.parameters(),
                                              lr=self.hyperparameters["Critic"]["learning_rate"], eps=5e-4)
        self.emb.eval()

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        if self.add_extra_noise: self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep:
                self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1
        print('step', self.episode_step_number_val - self.environment._max_episode_steps)
        print(self.total_episode_score_so_far)
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

    def convert_state(self, state):
        state = torch.tensor(state).view(1, -1, 1).to(torch.float32)
        if len(self.stacked_obs) == 0:
            for _ in range(self.n_history):
                self.stacked_obs.append(state)
        else:
            self.stacked_obs.pop(0)
            self.stacked_obs.append(state)
        return self.stacked_obs

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True
         3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        # print('eval_ep', eval_ep)
        # print('self.global_step_number', self.global_step_number)
        if state is None: state = self.state
        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)

        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = self.environment.action_space.sample()
            # print("Picking random action ", action)
        else:
            action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None:
            state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if eval is False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        state = self.convert_state(state)
        emb_input = torch.cat(state, dim=-1)
        state = self.emb(emb_input, self.adj).detach()
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample means it is sampled using reparameterisation trick
        # TODO TZM modify here
        action = torch.tanh(x_t)
        # action = x_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def my_produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        # state = self.convert_state(state)
        # emb_input = torch.cat(state, dim=-1)
        # adj = torch.cat([self.adj for _ in range(self.hyperparameters["batch_size"])], dim=0).to(torch.float32)
        # emb_input = state.unsqueeze(-1)
        # state = self.emb(emb_input, adj).detach()
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample means it is sampled using reparameterisation trick
        # TODO TZM modify here
        action = torch.tanh(x_t)
        # action = x_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
               self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
                   "update_every_n_steps"] == 0

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
                                                          mask_batch)
        self.update_critic_parameters(qf1_loss, qf2_loss)
        self.emb.train()
        # qf_loss = qf1_loss + qf2_loss
        # self.emb_optimizer.zero_grad()
        # qf_loss.backward()
        # self.emb_optimizer.step()

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        # self.emb_optimizer.zero_grad()
        # qf_loss.backward()
        # self.emb_optimizer.step()
        # self.emb.eval()
        # self.soft_update_of_target_network(self.emb, self.temb, self.hyperparameters["Critic"]["tau"])

        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_actor_parameters(policy_loss, alpha_loss)

    def sample_experiences(self):
        return self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        adj = torch.cat([self.adj for _ in range(self.hyperparameters["batch_size"])], dim=0).to(torch.float32)
        state_batch = self.emb(state_batch.unsqueeze(-1), adj).detach()
        next_state_batch = self.emb(next_state_batch.unsqueeze(-1), adj).detach()
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.my_produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(torch.cat((next_state_batch, next_state_action), 1))
            qf2_next_target = self.critic_target_2(torch.cat((next_state_batch, next_state_action), 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)
        qf1 = self.critic_local(torch.cat((state_batch, action_batch), 1))
        qf2 = self.critic_local_2(torch.cat((state_batch, action_batch), 1))
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.my_produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(torch.cat((state_batch, action), 1))
        qf2_pi = self.critic_local_2(torch.cat((state_batch, action), 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter.
        This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_critic_parameters(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for both critics"""
        # self.my_take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
        #                                self.hyperparameters["Critic"]["gradient_clipping_norm"])
        # self.my_take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
        #                                self.hyperparameters["Critic"]["gradient_clipping_norm"])

        loss = critic_loss_1 + critic_loss_2
        self.critic_optimizer.zero_grad()
        self.emb_optimizer.zero_grad()
        loss.backward()
        self.emb_optimizer.step()
        self.critic_optimizer.step()

        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        # self.my_take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
        #                                self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.emb_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.emb_optimizer.step()
        self.actor_optimizer.step()
        if alpha_loss is not None:
            self.my_take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def my_take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        loss.backward(retain_graph=retain_graph)  # this calculates the gradients
        self.logger.info("Loss -- {}".format(loss.item()))
        if self.debug_mode:
            print('debug mode')
            for net in network:
                self.log_gradient_and_weight_information(net, optimizer)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               clipping_norm)  # clip gradients to help stabilise training
        # optimizer.step()  # this applies the gradients

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")

    def locally_save_policy(self, filename=None):
        """save model and optimizers

        Details：...

        Args:
             filename (str): file name
             self

        Returns:
            None
        """

        if filename is None:
            filename = "./models_try_00/"
            if not os.path.exists("./models_try_00"):
                os.makedirs("./models_try_00")

        filename += 'SAC'
        filename += str(id(self))

        torch.save(self.critic_local.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor_local.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename, id_input):
        """load model and optimizers

        Details：...

        Args:
             filename (str): file name
             id_input (str): id number
             self

        Returns:
            None
        """
        filename += id_input
        print('SAC 00000000000000000000000000000000000000000000001111')

        self.critic_local.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic_local)

        self.actor_local.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor_local)
