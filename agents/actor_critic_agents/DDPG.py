import os
import copy
import torch
import torch.nn.functional as functional
from torch import optim
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
import pickle


class DDPG(Base_Agent):
    """A DDPG Agent"""
    agent_name = "DDPG"

    def __init__(self, config):
        print("-----------------------------------------------------------------------------------")
        Base_Agent.__init__(self, config)
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic")
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        Base_Agent.copy_model_over(self.actor_local, self.actor_target)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.exploration_strategy = OU_Noise_Exploration(self.config)
        if not self.config.hyperparameters["train_model"]:
            self.load("./models_try_00/", self.config.hyperparameters["load_model_id"])
            # pass

    def step(self):
        """Runs a step in the game"""
        replay_record = {"action": [], "state": [], "real_action": [], "real_state": []}
        # replay_record[]self.state
        # self.action
        while not self.done:
            # print("State ", self.state.shape)
            # print(999)
            self.action = self.pick_action()
            # print("action", self.action)
            self.conduct_action(self.action)
            # print("state", self.environment.env.real_state)
            replay_record["action"].append(self.action)
            replay_record["state"].append(self.state)
            replay_record["real_action"].append(self.environment.env.action_real)
            replay_record["real_state"].append(self.environment.env.real_state)
            # replay_record["real_action"].append(self.environment.env)
            if self.time_for_critic_and_actor_to_learn() and self.config.hyperparameters["train_model"]:
                # print('###################################start')
                # print('id(self)', id(self))
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.sample_experiences()
                    self.critic_learn(states, actions, rewards, next_states, dones)
                    self.actor_learn(states)
                # print('###################################end')
            self.save_experience()
            self.state = self.next_state  # this is to set the state for the next iteration
            self.global_step_number += 1
        file_name = 'replay_record' + str(self.episode_number) + '.pkl'
        if not self.hyperparameters['train_model']:
            with open(file_name, 'wb') as fp:
                pickle.dump(replay_record, fp)
        self.episode_number += 1

    def sample_experiences(self):
        return self.memory.sample()

    def pick_action(self, state=None):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        if state is None:
            # print('pick_action_state', self.state)
            # print('type_state', type(self.state))
            state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})
        action = action.squeeze(0)
        action = action.clip(-1, 1)
        return action

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for the critic"""
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss for the critic"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = functional.mse_loss(critic_expected, critic_targets)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            critic_targets_next = self.critic_target(torch.cat((next_states, actions_next), 1))
        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):
        """Computes the expected critic values to be used in the loss for the critic"""
        critic_expected = self.critic_local(torch.cat((states, actions), 1))
        return critic_expected

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
            "update_every_n_steps"] == 0 and self.hyperparameters["train_model"]

    def actor_learn(self, states):
        """Runs a learning iteration for the actor"""
        if self.done:  # we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
        actor_loss = self.calculate_actor_loss(states)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])

    def calculate_actor_loss(self, states):
        """Calculates the loss for the actor"""
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        return actor_loss

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

        filename += 'DDPG'
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
        print('00000000000000000000000000000000000000000000001111')

        self.critic_local.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic_local)

        self.actor_local.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor_local)
