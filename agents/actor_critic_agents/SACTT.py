import os
import sys
import copy
import pickle
from agents.Base_Agent import Base_Agent
from utilities.OU_Noise import OU_Noise
from utilities.data_structures.Replay_Buffer import Replay_Buffer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import GATNNA, GATNNQ
from models import GOTNNA, GOTNNQ
from models import ATNNA, ATNNQ
from models import TNNA, TNNQ
from models import UNTNNA, UNTNNQ
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class TTSAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github
    implementation https://github.com/pranz24/pytorch-soft-actor-critic.
    It is an actor-critic algorithm where the agent is also trained
    to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SACTT"

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
        self.time_window = self.hyperparameters["time_window"]
        self.emb_dim = 64
        self.nheads = 8

        print("state_size", self.state_size)
        print("action_size", self.action_size)

        aa = self.state_size + self.action_size
        self.adj = torch.FloatTensor(np.ones((self.state_size, self.state_size))).unsqueeze(0)
        self.tt = True  # use gated attention
        if self.tt:
            self.critic_local = GATNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size, output_dim=1,
                                       mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                       nheads=self.nheads, adj_base=self.adj).to(self.device)
            self.critic_local_2 = GATNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                         output_dim=1,
                                         mask_dim=self.state_size,
                                         emb_dim=self.emb_dim, window_size=self.time_window,
                                         nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                               key_to_use="Critic")
            self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                 key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        if self.tt:
            self.critic_target = GATNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                        output_dim=1,
                                        mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                        nheads=self.nheads, adj_base=self.adj).to(self.device)
            self.critic_target_2 = GATNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                          output_dim=1,
                                          mask_dim=self.state_size,
                                          emb_dim=self.emb_dim, window_size=self.time_window,
                                          nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                key_to_use="Critic")
            self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                  key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)
        if self.tt:
            self.actor_local = GATNNA(input_dim=self.state_size, output_dim=self.action_size * 2,
                                      mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                      nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.actor_local = self.create_NN(input_dim=self.state_size,
                                              output_dim=self.action_size * 2,
                                              key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-5)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-5)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        if not self.config.hyperparameters["train_model"]:
            self.load("./models_try_00/", self.config.hyperparameters["load_model_id"])

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self.attached_states = None

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        self.attached_states = None
        if self.add_extra_noise: self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        # noinspection PyAttributeOutsideInit
        self.episode_step_number_val = 0
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": []}
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)

            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)

            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            # self.state_save=self.attached_states
            # state_temp = torch.cat([self.attached_states, state.unsqueeze(-1)], dim=-1)
            # _, state_temp = state_temp.split([1, self.time_window], dim=-1)
            # self.next_state_save=
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep:
                next_state_vector = torch.FloatTensor([self.next_state]).to(self.device)
                if len(next_state_vector.shape) == 1:
                    next_state_vector = next_state_vector.unsqueeze(0)
                next_state_vector = torch.cat([self.attached_states, next_state_vector.unsqueeze(-1)], dim=-1)
                _, next_state_vector = next_state_vector.split([1, self.time_window], dim=-1)
                self.save_experience(
                    experience=(self.attached_states.cpu(), self.action, self.reward, next_state_vector.cpu(), mask))
            self.state = self.next_state
            self.global_step_number += 1
        print('step', self.episode_step_number_val - self.environment._max_episode_steps)
        print(self.total_episode_score_so_far)
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        file_name = 'replay_record_1023' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def step_test(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": [], "charge_power": [],
                         "car_number": [], "flow_in_number": [], "load_assigned": [], "min_power": [], "max_power": []}
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.actor_pick_action(state=self.state, eval=True)
            self.conduct_action(self.action)

            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)
            replay_record["charge_power"].append(self.environment.env_aggregator.evcssp_charge_power)
            replay_record["car_number"].append(self.environment.env_aggregator.ag_car_number)
            replay_record["flow_in_number"].append(self.environment.env_aggregator.ag_flow_in_number)
            replay_record["load_assigned"].append(self.environment.env_aggregator.ag_load_assigned)
            replay_record["min_power"].append(self.environment.env_aggregator.evcssp_min_demand)
            replay_record["max_power"].append(self.environment.env_aggregator.evcssp_max_demand)

            self.state = self.next_state
            self.global_step_number += 1

        file_name = 'replay_record_test_' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True
         3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        # print('eval_ep', eval_ep)
        # print('self.global_step_number', self.global_step_number)
        if state is None:
            state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        self.stack_state(state)
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
        assert state is not None, 'state can not be NONE'
        assert len(state.shape) != 1, 'state shape error'

        if eval is False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state, use_exp=False):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        if use_exp:
            actor_output = self.actor_local(state)
        else:
            actor_output = self.actor_local(self.attached_states)
        # actor_output = self.actor_local(state)
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

    def stack_state(self, state):
        if self.attached_states is None:
            state_temp = state.unsqueeze(-1).expand(-1, -1, self.time_window)
        else:
            state_temp = torch.cat([self.attached_states, state.unsqueeze(-1)], dim=-1)
            _, state_temp = state_temp.split([1, self.time_window], dim=-1)
        self.attached_states = state_temp
        assert self.attached_states.shape[2] == self.time_window, 'state length error! time window'
        assert self.attached_states.shape[1] == self.state_size, 'state length error! state dim'

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

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
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
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch,
                                                                                          use_exp=True)
            # TODO 1
            qf1_next_target = self.critic_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_target_2(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)
        qf1 = self.critic_local(state_batch, action_batch)
        qf2 = self.critic_local_2(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch, use_exp=True)
        qf1_pi = self.critic_local(state_batch, action)
        qf2_pi = self.critic_local_2(state_batch, action)
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
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

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

        filename += 'SACTT'
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
        filename += "SACTT"
        filename += id_input
        print('SAC 00000000000000000000000000000000000000000000001111')

        self.critic_local.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic_local)

        self.actor_local.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor_local)


class OOSAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github
    implementation https://github.com/pranz24/pytorch-soft-actor-critic.
    It is an actor-critic algorithm where the agent is also trained
    to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SACTT_wo_at"

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
        self.time_window = self.hyperparameters["time_window"]
        self.emb_dim = 64
        self.nheads = 8

        print("state_size", self.state_size)
        print("action_size", self.action_size)

        aa = self.state_size + self.action_size
        self.adj = torch.FloatTensor(np.ones((self.state_size, self.state_size))).unsqueeze(0)
        self.tt = True  # use gated attention
        if self.tt:
            self.critic_local = GOTNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size, output_dim=1,
                                       mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                       nheads=self.nheads, adj_base=self.adj).to(self.device)
            self.critic_local_2 = GOTNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                         output_dim=1,
                                         mask_dim=self.state_size,
                                         emb_dim=self.emb_dim, window_size=self.time_window,
                                         nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                               key_to_use="Critic")
            self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                 key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        if self.tt:
            self.critic_target = GOTNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                        output_dim=1,
                                        mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                        nheads=self.nheads, adj_base=self.adj).to(self.device)
            self.critic_target_2 = GOTNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                          output_dim=1,
                                          mask_dim=self.state_size,
                                          emb_dim=self.emb_dim, window_size=self.time_window,
                                          nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                key_to_use="Critic")
            self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                  key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)
        if self.tt:
            self.actor_local = GOTNNA(input_dim=self.state_size, output_dim=self.action_size * 2,
                                      mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                      nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.actor_local = self.create_NN(input_dim=self.state_size,
                                              output_dim=self.action_size * 2,
                                              key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-5)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-5)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        if not self.config.hyperparameters["train_model"]:
            self.load("./models_try_00/", self.config.hyperparameters["load_model_id"])

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self.attached_states = None

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        self.attached_states = None
        if self.add_extra_noise: self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        # noinspection PyAttributeOutsideInit
        self.episode_step_number_val = 0
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": []}
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)

            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)

            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            # self.state_save=self.attached_states
            # state_temp = torch.cat([self.attached_states, state.unsqueeze(-1)], dim=-1)
            # _, state_temp = state_temp.split([1, self.time_window], dim=-1)
            # self.next_state_save=
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep:
                next_state_vector = torch.FloatTensor([self.next_state]).to(self.device)
                if len(next_state_vector.shape) == 1:
                    next_state_vector = next_state_vector.unsqueeze(0)
                next_state_vector = torch.cat([self.attached_states, next_state_vector.unsqueeze(-1)], dim=-1)
                _, next_state_vector = next_state_vector.split([1, self.time_window], dim=-1)
                self.save_experience(
                    experience=(self.attached_states.cpu(), self.action, self.reward, next_state_vector.cpu(), mask))
            self.state = self.next_state
            self.global_step_number += 1
        print('step', self.episode_step_number_val - self.environment._max_episode_steps)
        print(self.total_episode_score_so_far)
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        file_name = 'replay_record_1023' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def step_test(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": [], "charge_power": [],
                         "car_number": [], "flow_in_number": [], "load_assigned": [], "min_power": [], "max_power": []}
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.actor_pick_action(state=self.state, eval=True)
            self.conduct_action(self.action)

            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)
            replay_record["charge_power"].append(self.environment.env_aggregator.evcssp_charge_power)
            replay_record["car_number"].append(self.environment.env_aggregator.ag_car_number)
            replay_record["flow_in_number"].append(self.environment.env_aggregator.ag_flow_in_number)
            replay_record["load_assigned"].append(self.environment.env_aggregator.ag_load_assigned)
            replay_record["min_power"].append(self.environment.env_aggregator.evcssp_min_demand)
            replay_record["max_power"].append(self.environment.env_aggregator.evcssp_max_demand)

            self.state = self.next_state
            self.global_step_number += 1

        file_name = 'replay_record_test_' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True
         3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        # print('eval_ep', eval_ep)
        # print('self.global_step_number', self.global_step_number)
        if state is None:
            state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        self.stack_state(state)
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
        assert state is not None, 'state can not be NONE'
        assert len(state.shape) != 1, 'state shape error'

        if eval is False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state, use_exp=False):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        if use_exp:
            actor_output = self.actor_local(state)
        else:
            actor_output = self.actor_local(self.attached_states)
        # actor_output = self.actor_local(state)
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

    def stack_state(self, state):
        if self.attached_states is None:
            state_temp = state.unsqueeze(-1).expand(-1, -1, self.time_window)
        else:
            state_temp = torch.cat([self.attached_states, state.unsqueeze(-1)], dim=-1)
            _, state_temp = state_temp.split([1, self.time_window], dim=-1)
        self.attached_states = state_temp
        assert self.attached_states.shape[2] == self.time_window, 'state length error! time window'
        assert self.attached_states.shape[1] == self.state_size, 'state length error! state dim'

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

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
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
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch,
                                                                                          use_exp=True)
            # TODO 1
            qf1_next_target = self.critic_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_target_2(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)
        qf1 = self.critic_local(state_batch, action_batch)
        qf2 = self.critic_local_2(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch, use_exp=True)
        qf1_pi = self.critic_local(state_batch, action)
        qf2_pi = self.critic_local_2(state_batch, action)
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
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

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

        filename += 'SACTT_wo_at'
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
        filename += "SACTT_wo_at"
        filename += id_input
        print('SAC 00000000000000000000000000000000000000000000001111')

        self.critic_local.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic_local)

        self.actor_local.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor_local)


class NGSAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github
    implementation https://github.com/pranz24/pytorch-soft-actor-critic.
    It is an actor-critic algorithm where the agent is also trained
    to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SACTT_wo_pf"

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
        self.time_window = self.hyperparameters["time_window"]
        self.emb_dim = 64
        self.nheads = 8

        print("state_size", self.state_size)
        print("action_size", self.action_size)

        aa = self.state_size + self.action_size
        self.adj = torch.FloatTensor(np.ones((self.state_size, self.state_size))).unsqueeze(0)
        self.tt = True  # use gated attention
        if self.tt:
            self.critic_local = ATNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size, output_dim=1,
                                      mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                      nheads=self.nheads, adj_base=self.adj).to(self.device)
            self.critic_local_2 = ATNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                        output_dim=1,
                                        mask_dim=self.state_size,
                                        emb_dim=self.emb_dim, window_size=self.time_window,
                                        nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                               key_to_use="Critic")
            self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                 key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        if self.tt:
            self.critic_target = ATNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                       output_dim=1,
                                       mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                       nheads=self.nheads, adj_base=self.adj).to(self.device)
            self.critic_target_2 = ATNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                         output_dim=1,
                                         mask_dim=self.state_size,
                                         emb_dim=self.emb_dim, window_size=self.time_window,
                                         nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                key_to_use="Critic")
            self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                  key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)
        if self.tt:
            self.actor_local = ATNNA(input_dim=self.state_size, output_dim=self.action_size * 2,
                                     mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                     nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.actor_local = self.create_NN(input_dim=self.state_size,
                                              output_dim=self.action_size * 2,
                                              key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-5)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-5)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        if not self.config.hyperparameters["train_model"]:
            self.load("./models_try_00/", self.config.hyperparameters["load_model_id"])

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self.attached_states = None

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        self.attached_states = None
        if self.add_extra_noise: self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        # noinspection PyAttributeOutsideInit
        self.episode_step_number_val = 0
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": []}
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)

            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)

            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            # self.state_save=self.attached_states
            # state_temp = torch.cat([self.attached_states, state.unsqueeze(-1)], dim=-1)
            # _, state_temp = state_temp.split([1, self.time_window], dim=-1)
            # self.next_state_save=
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep:
                next_state_vector = torch.FloatTensor([self.next_state]).to(self.device)
                if len(next_state_vector.shape) == 1:
                    next_state_vector = next_state_vector.unsqueeze(0)
                next_state_vector = torch.cat([self.attached_states, next_state_vector.unsqueeze(-1)], dim=-1)
                _, next_state_vector = next_state_vector.split([1, self.time_window], dim=-1)
                self.save_experience(
                    experience=(self.attached_states.cpu(), self.action, self.reward, next_state_vector.cpu(), mask))
            self.state = self.next_state
            self.global_step_number += 1
        print('step', self.episode_step_number_val - self.environment._max_episode_steps)
        print(self.total_episode_score_so_far)
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        file_name = 'replay_record_1023' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def step_test(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": [], "charge_power": [],
                         "car_number": [], "flow_in_number": [], "load_assigned": [], "min_power": [], "max_power": []}
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.actor_pick_action(state=self.state, eval=True)
            self.conduct_action(self.action)

            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)
            replay_record["charge_power"].append(self.environment.env_aggregator.evcssp_charge_power)
            replay_record["car_number"].append(self.environment.env_aggregator.ag_car_number)
            replay_record["flow_in_number"].append(self.environment.env_aggregator.ag_flow_in_number)
            replay_record["load_assigned"].append(self.environment.env_aggregator.ag_load_assigned)
            replay_record["min_power"].append(self.environment.env_aggregator.evcssp_min_demand)
            replay_record["max_power"].append(self.environment.env_aggregator.evcssp_max_demand)

            self.state = self.next_state
            self.global_step_number += 1

        file_name = 'replay_record_test_' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True
         3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        # print('eval_ep', eval_ep)
        # print('self.global_step_number', self.global_step_number)
        if state is None:
            state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        self.stack_state(state)
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
        assert state is not None, 'state can not be NONE'
        assert len(state.shape) != 1, 'state shape error'

        if eval is False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state, use_exp=False):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        if use_exp:
            actor_output = self.actor_local(state)
        else:
            actor_output = self.actor_local(self.attached_states)
        # actor_output = self.actor_local(state)
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

    def stack_state(self, state):
        if self.attached_states is None:
            state_temp = state.unsqueeze(-1).expand(-1, -1, self.time_window)
        else:
            state_temp = torch.cat([self.attached_states, state.unsqueeze(-1)], dim=-1)
            _, state_temp = state_temp.split([1, self.time_window], dim=-1)
        self.attached_states = state_temp
        assert self.attached_states.shape[2] == self.time_window, 'state length error! time window'
        assert self.attached_states.shape[1] == self.state_size, 'state length error! state dim'

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

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
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
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch,
                                                                                          use_exp=True)
            # TODO 1
            qf1_next_target = self.critic_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_target_2(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)
        qf1 = self.critic_local(state_batch, action_batch)
        qf2 = self.critic_local_2(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch, use_exp=True)
        qf1_pi = self.critic_local(state_batch, action)
        qf2_pi = self.critic_local_2(state_batch, action)
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
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

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

        filename += '"SACTT_wo_pf"'
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
        filename += "SACTT_wo_at"
        filename += id_input
        print('SAC 00000000000000000000000000000000000000000000001111')

        self.critic_local.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic_local)

        self.actor_local.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor_local)


class NNSAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github
    implementation https://github.com/pranz24/pytorch-soft-actor-critic.
    It is an actor-critic algorithm where the agent is also trained
    to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SACTT_wo_gt"

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
        self.time_window = self.hyperparameters["time_window"]
        self.emb_dim = 64
        self.nheads = 8

        print("state_size", self.state_size)
        print("action_size", self.action_size)

        aa = self.state_size + self.action_size
        self.adj = torch.FloatTensor(np.ones((self.state_size, self.state_size))).unsqueeze(0)
        self.tt = True  # use gated attention
        if self.tt:
            self.critic_local = TNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size, output_dim=1,
                                     mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                     nheads=self.nheads, adj_base=self.adj).to(self.device)
            self.critic_local_2 = TNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                       output_dim=1,
                                       mask_dim=self.state_size,
                                       emb_dim=self.emb_dim, window_size=self.time_window,
                                       nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                               key_to_use="Critic")
            self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                 key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        if self.tt:
            self.critic_target = TNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                      output_dim=1,
                                      mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                      nheads=self.nheads, adj_base=self.adj).to(self.device)
            self.critic_target_2 = TNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                        output_dim=1,
                                        mask_dim=self.state_size,
                                        emb_dim=self.emb_dim, window_size=self.time_window,
                                        nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                key_to_use="Critic")
            self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                  key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)
        if self.tt:
            self.actor_local = TNNA(input_dim=self.state_size, output_dim=self.action_size * 2,
                                    mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                    nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.actor_local = self.create_NN(input_dim=self.state_size,
                                              output_dim=self.action_size * 2,
                                              key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-5)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-5)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        if not self.config.hyperparameters["train_model"]:
            self.load("./models_try_00/", self.config.hyperparameters["load_model_id"])

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self.attached_states = None

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        self.attached_states = None
        if self.add_extra_noise: self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        # noinspection PyAttributeOutsideInit
        self.episode_step_number_val = 0
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": []}
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)

            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)

            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            # self.state_save=self.attached_states
            # state_temp = torch.cat([self.attached_states, state.unsqueeze(-1)], dim=-1)
            # _, state_temp = state_temp.split([1, self.time_window], dim=-1)
            # self.next_state_save=
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep:
                next_state_vector = torch.FloatTensor([self.next_state]).to(self.device)
                if len(next_state_vector.shape) == 1:
                    next_state_vector = next_state_vector.unsqueeze(0)
                next_state_vector = torch.cat([self.attached_states, next_state_vector.unsqueeze(-1)], dim=-1)
                _, next_state_vector = next_state_vector.split([1, self.time_window], dim=-1)
                self.save_experience(
                    experience=(self.attached_states.cpu(), self.action, self.reward, next_state_vector.cpu(), mask))
            self.state = self.next_state
            self.global_step_number += 1
        print('step', self.episode_step_number_val - self.environment._max_episode_steps)
        print(self.total_episode_score_so_far)
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        file_name = 'replay_record_1023' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def step_test(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": [], "charge_power": [],
                         "car_number": [], "flow_in_number": [], "load_assigned": [], "min_power": [], "max_power": []}
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.actor_pick_action(state=self.state, eval=True)
            self.conduct_action(self.action)

            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)
            replay_record["charge_power"].append(self.environment.env_aggregator.evcssp_charge_power)
            replay_record["car_number"].append(self.environment.env_aggregator.ag_car_number)
            replay_record["flow_in_number"].append(self.environment.env_aggregator.ag_flow_in_number)
            replay_record["load_assigned"].append(self.environment.env_aggregator.ag_load_assigned)
            replay_record["min_power"].append(self.environment.env_aggregator.evcssp_min_demand)
            replay_record["max_power"].append(self.environment.env_aggregator.evcssp_max_demand)

            self.state = self.next_state
            self.global_step_number += 1

        file_name = 'replay_record_test_' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True
         3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        # print('eval_ep', eval_ep)
        # print('self.global_step_number', self.global_step_number)
        if state is None:
            state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        self.stack_state(state)
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
        assert state is not None, 'state can not be NONE'
        assert len(state.shape) != 1, 'state shape error'

        if eval is False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state, use_exp=False):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        if use_exp:
            actor_output = self.actor_local(state)
        else:
            actor_output = self.actor_local(self.attached_states)
        # actor_output = self.actor_local(state)
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

    def stack_state(self, state):
        if self.attached_states is None:
            state_temp = state.unsqueeze(-1).expand(-1, -1, self.time_window)
        else:
            state_temp = torch.cat([self.attached_states, state.unsqueeze(-1)], dim=-1)
            _, state_temp = state_temp.split([1, self.time_window], dim=-1)
        self.attached_states = state_temp
        assert self.attached_states.shape[2] == self.time_window, 'state length error! time window'
        assert self.attached_states.shape[1] == self.state_size, 'state length error! state dim'

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

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
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
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch,
                                                                                          use_exp=True)
            # TODO 1
            qf1_next_target = self.critic_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_target_2(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)
        qf1 = self.critic_local(state_batch, action_batch)
        qf2 = self.critic_local_2(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch, use_exp=True)
        qf1_pi = self.critic_local(state_batch, action)
        qf2_pi = self.critic_local_2(state_batch, action)
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
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

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

        filename += 'SACNN'
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
        filename += "SACNN"
        filename += id_input
        print('SAC 00000000000000000000000000000000000000000000001111')

        self.critic_local.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic_local)

        self.actor_local.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor_local)


class UTSAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github
    implementation https://github.com/pranz24/pytorch-soft-actor-critic.
    It is an actor-critic algorithm where the agent is also trained
    to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SACT_Uni"

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
        self.time_window = self.hyperparameters["time_window"]
        self.emb_dim = 64
        self.nheads = 8

        print("state_size", self.state_size)
        print("action_size", self.action_size)

        aa = self.state_size + self.action_size
        self.adj = torch.FloatTensor(np.ones((self.state_size, self.state_size))).unsqueeze(0)
        self.tt = True  # use gated attention
        if self.tt:
            self.critic_local = UNTNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size, output_dim=1,
                                       mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                       nheads=self.nheads, adj_base=self.adj).to(self.device)
            self.critic_local_2 = UNTNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                         output_dim=1,
                                         mask_dim=self.state_size,
                                         emb_dim=self.emb_dim, window_size=self.time_window,
                                         nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                               key_to_use="Critic")
            self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                 key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        if self.tt:
            self.critic_target = UNTNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                        output_dim=1,
                                        mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                        nheads=self.nheads, adj_base=self.adj).to(self.device)
            self.critic_target_2 = UNTNNQ(input_state_dim=self.state_size, input_action_dim=self.action_size,
                                          output_dim=1,
                                          mask_dim=self.state_size,
                                          emb_dim=self.emb_dim, window_size=self.time_window,
                                          nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                key_to_use="Critic")
            self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                                  key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)
        if self.tt:
            self.actor_local = UNTNNA(input_dim=self.state_size, output_dim=self.action_size * 2,
                                      mask_dim=self.state_size, emb_dim=self.emb_dim, window_size=self.time_window,
                                      nheads=self.nheads, adj_base=self.adj).to(self.device)
        else:
            self.actor_local = self.create_NN(input_dim=self.state_size,
                                              output_dim=self.action_size * 2,
                                              key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-5)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-5)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        if not self.config.hyperparameters["train_model"]:
            self.load("./models_try_00/", self.config.hyperparameters["load_model_id"])

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self.attached_states = None

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        self.attached_states = None
        if self.add_extra_noise: self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        # noinspection PyAttributeOutsideInit
        self.episode_step_number_val = 0
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": []}
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)

            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)

            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            # self.state_save=self.attached_states
            # state_temp = torch.cat([self.attached_states, state.unsqueeze(-1)], dim=-1)
            # _, state_temp = state_temp.split([1, self.time_window], dim=-1)
            # self.next_state_save=
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep:
                next_state_vector = torch.FloatTensor([self.next_state]).to(self.device)
                if len(next_state_vector.shape) == 1:
                    next_state_vector = next_state_vector.unsqueeze(0)
                next_state_vector = torch.cat([self.attached_states, next_state_vector.unsqueeze(-1)], dim=-1)
                _, next_state_vector = next_state_vector.split([1, self.time_window], dim=-1)
                self.save_experience(
                    experience=(self.attached_states.cpu(), self.action, self.reward, next_state_vector.cpu(), mask))
            self.state = self.next_state
            self.global_step_number += 1
        print('step', self.episode_step_number_val - self.environment._max_episode_steps)
        print(self.total_episode_score_so_far)
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        file_name = 'replay_record_1023' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def step_test(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": [], "charge_power": [],
                         "car_number": [], "flow_in_number": [], "load_assigned": [], "min_power": [], "max_power": []}
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.actor_pick_action(state=self.state, eval=True)
            self.conduct_action(self.action)

            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)
            replay_record["charge_power"].append(self.environment.env_aggregator.evcssp_charge_power)
            replay_record["car_number"].append(self.environment.env_aggregator.ag_car_number)
            replay_record["flow_in_number"].append(self.environment.env_aggregator.ag_flow_in_number)
            replay_record["load_assigned"].append(self.environment.env_aggregator.ag_load_assigned)
            replay_record["min_power"].append(self.environment.env_aggregator.evcssp_min_demand)
            replay_record["max_power"].append(self.environment.env_aggregator.evcssp_max_demand)

            self.state = self.next_state
            self.global_step_number += 1

        file_name = 'replay_record_test_' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True
         3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        # print('eval_ep', eval_ep)
        # print('self.global_step_number', self.global_step_number)
        if state is None:
            state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        self.stack_state(state)
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
        assert state is not None, 'state can not be NONE'
        assert len(state.shape) != 1, 'state shape error'

        if eval is False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state, use_exp=False):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        if use_exp:
            actor_output = self.actor_local(state)
        else:
            actor_output = self.actor_local(self.attached_states)
        # actor_output = self.actor_local(state)
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

    def stack_state(self, state):
        if self.attached_states is None:
            state_temp = state.unsqueeze(-1).expand(-1, -1, self.time_window)
        else:
            state_temp = torch.cat([self.attached_states, state.unsqueeze(-1)], dim=-1)
            _, state_temp = state_temp.split([1, self.time_window], dim=-1)
        self.attached_states = state_temp
        assert self.attached_states.shape[2] == self.time_window, 'state length error! time window'
        assert self.attached_states.shape[1] == self.state_size, 'state length error! state dim'

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

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
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
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch,
                                                                                          use_exp=True)
            # TODO 1
            qf1_next_target = self.critic_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_target_2(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)
        qf1 = self.critic_local(state_batch, action_batch)
        qf2 = self.critic_local_2(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch, use_exp=True)
        qf1_pi = self.critic_local(state_batch, action)
        qf2_pi = self.critic_local_2(state_batch, action)
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
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

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

        filename += 'SACT_Uni'
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
        filename += "SACT_Uni"
        filename += id_input
        print('SAC 00000000000000000000000000000000000000000000001111')

        self.critic_local.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic_local)

        self.actor_local.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor_local)
