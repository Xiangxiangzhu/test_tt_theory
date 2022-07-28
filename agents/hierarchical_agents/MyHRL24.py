import copy
import torch
import numpy as np
from gym import Wrapper
from agents.Base_Agent import Base_Agent
from agents.actor_critic_agents.DDPG import DDPG


class EVCSSPHRL_v2(Base_Agent):
    agent_name = "EVCSSPHRL"

    def __init__(self, config):
        super().__init__(config)
        # maximum time steps the lower agent takes
        self.max_sub_policy_timesteps = config.hyperparameters["LOWER_LEVEL"]["max_lower_level_timesteps"]

        self.config.hyperparameters = self.config.hyperparameters

        # the observation and next observation of upper agent
        self.higher_level_state = None  # true state of environment
        self.higher_level_next_state = None

        # reward of both agents
        self.higher_level_reward = None
        self.lower_level_reward = None

        # done flag of both agents
        self.higher_level_done = False
        self.lower_level_done = False

        # the upper agent's action
        self.goal = None

        # the observation and next observation of lower agent
        self.lower_level_state = None  # state of environment with goal appended
        self.lower_level_next_state = None

        # set lower level agent configuration
        self.lower_level_agent_config = copy.copy(config)
        self.lower_level_agent_config.hyperparameters = self.lower_level_agent_config.hyperparameters["LOWER_LEVEL"]

        # set lower level agent environment
        self.lower_level_agent_config.environment = Lower_Level_Agent_Environment_Wrapper(self.environment, self,
                                                                                          self.max_sub_policy_timesteps)
        # agent of lower level
        # TODO: change agent type
        self.lower_level_agent = DDPG(self.lower_level_agent_config)

        # aimed reward of lower level agent
        self.lower_level_agent.average_score_required_to_win = float("inf")

        # set the upper level agent configuration
        self.higher_level_agent_config = copy.copy(config)
        self.higher_level_agent_config.hyperparameters = self.higher_level_agent_config.hyperparameters["HIGHER_LEVEL"]

        # set upper level agent environment
        self.higher_level_agent_config.environment = Higher_Level_Agent_Environment_Wrapper(self.environment, self)

        # set higher level action size
        self.higher_level_agent_config.overwrite_action_size = 1,

        # agent of higher level
        # TODO: change agent type
        self.higher_level_agent = HRL_Higher_Level_DRL_Agent(self.higher_level_agent_config,
                                                             self.lower_level_agent.actor_local)

        # for method "save_higher_level_experience" to store
        self.step_lower_level_states = []
        self.step_lower_level_action_seen = []

    def run_n_episodes(self, **kwargs):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        a, b, c = self.higher_level_agent.run_n_episodes(self.config.num_episodes_to_run)
        return a, b, c

    @staticmethod
    def goal_transition(goal):
        """Provides updated goal according to the goal transition function in the HIRO paper"""
        return goal

    def save_higher_level_experience(self):
        self.higher_level_agent.step_lower_level_states = self.step_lower_level_states
        self.higher_level_agent.step_lower_level_action_seen = self.step_lower_level_action_seen


class HRL_Higher_Level_DRL_Agent(DDPG):
    """Extends DDPG so that it can function as the higher level agent in the HIRO hierarchical RL algorithm.
    This only involves changing how the agent saves experiences and samples them for learning"""

    def __init__(self, config, lower_level_policy):
        super(HRL_Higher_Level_DRL_Agent, self).__init__(config)
        # the policy network of lower agent
        self.step_lower_level_states = None
        self.step_lower_level_action_seen = None
        self.lower_level_policy = lower_level_policy

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer. Adapted from normal DDPG so that it saves the sequence of
        states, goals and actions that we saw whilst control was given to the lower level"""
        if memory is None:
            memory = self.memory
        if experience is None:
            # experience = self.step_lower_level_states, self.step_lower_level_action_seen, self.reward, self.next_state, self.done
            # print('self.reward', self.reward)
            experience = self.state, self.action, self.reward, self.next_state, self.done
        else:
            print(1234567812345678)
        # print('higher_experience', id(self), experience)
        memory.add_experience(*experience)

    def sample_experiences(self):
        # experiences = self.memory.produce_action_and_action_info(separate_out_data_types=False)
        # print('inter sample_experiences')
        # print('******************************************')
        experiences = self.memory.sample(separate_out_data_types=False)
        # assert len(experiences[0].state) == self.hyperparameters["max_lower_level_timesteps"] or experiences[0].done
        # assert experiences[0].state[0].shape[0] == self.state_size * 2
        # assert len(experiences[0].action) == self.hyperparameters["max_lower_level_timesteps"] or experiences[0].done

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for ix, experience in enumerate(experiences):
            # state, action, reward, next_state, done = self.transform_goal_to_one_most_likely_to_have_induced_actions(
            #     experience)
            state, action, reward, next_state, done = self.transform_experience_to_my_training_data(experience)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.from_numpy(np.vstack([state for state in states])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([action for action in actions])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([reward for reward in rewards])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([next_state for next_state in next_states])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(done) for done in dones])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def transform_experience_to_my_training_data(self, experience):
        assert self is not None
        # state = experience.state[0][:self.state_size]
        state = experience.state
        next_state = experience.next_state
        reward = experience.reward
        action = experience.action
        # action = experience.state[0][self.state_size:]
        done = experience.done
        return state, action, reward, next_state, done


class Higher_Level_Agent_Environment_Wrapper(Wrapper):
    """Adapts the game environment so that it is compatible with the higher level agent which sets goals for the lower
    level agent"""

    def __init__(self, env, HIRO_agent):
        Wrapper.__init__(self, env)
        self.env = env
        self.HIRO_agent = HIRO_agent
        self.action_space = self.observation_space
        print('init higher level agent env wrapper')

    def reset(self, **kwargs):
        # print('inter reset method in higher level agent env wrapper')
        # self.count = 0
        all_env_state = self.env.reset(**kwargs)
        # TODO: 更新状态
        # self.HIRO_agent.higher_level_state = all_env_state[:5]
        self.HIRO_agent.higher_level_state = all_env_state[[0, 1, 2, 4]]

        return self.HIRO_agent.higher_level_state

    def step(self, goal):
        # print('inter step method in higher level agent env wrapper')
        # print('self.count', self.count)
        # self.count += 1

        print('')
        print('set_goal')

        self.HIRO_agent.higher_level_reward = 0
        self.HIRO_agent.step_lower_level_states = []
        self.HIRO_agent.step_lower_level_action_seen = []

        self.HIRO_agent.goal = goal

        # get real goal
        self.HIRO_agent.real_goal = self.env.to_real_goal(goal)

        self.env.env_aggregator.consult_top_agent(self.HIRO_agent.real_goal)

        self.HIRO_agent.lower_level_agent.episode_number = 0  # must reset lower level agent to 0 episodes completed otherwise won't run more episodes
        self.HIRO_agent.lower_level_agent.run_n_episodes(num_episodes=1, show_whether_achieved_goal=False,
                                                         save_and_print_results=True)

        self.HIRO_agent.save_higher_level_experience()

        return self.HIRO_agent.higher_level_next_state, self.HIRO_agent.higher_level_reward, self.HIRO_agent.higher_level_done, {}


class Lower_Level_Agent_Environment_Wrapper(Wrapper):
    """Open AI gym wrapper to help create an environment where a goal from a higher-level agent is treated as part
    of the environment state"""

    def __init__(self, env, HIRO_agent, max_sub_policy_timesteps):
        Wrapper.__init__(self, env)
        self.env = env
        self.meta_agent = HIRO_agent
        self.max_sub_policy_timesteps = max_sub_policy_timesteps
        self.lower_level_timesteps = 0
        self.track_intrinsic_rewards = []
        print('init lower level agent env wrapper')

    def reset(self, **kwargs):
        # conduct this progress every ep
        # print('inter reset method in lower level agent env wrapper')
        # self.lower_count = 0
        if self.meta_agent.higher_level_state is not None:
            # state = self.meta_agent.higher_level_state
            state = self.env.state
            # print('set goal, receive higher level agent state')
            assert len(state) == self.env.observation_space.shape[0], 'state length error3'
        else:
            print("INITIATION ONLY")
            state = self.env.reset()
            print('self.env.observation_space.shape', self.env.observation_space.shape[0])
            assert len(state) == self.env.observation_space.shape[0], 'state length error4'

        if self.meta_agent.goal is not None:
            print('to set goal now')
            real_goal = self.meta_agent.real_goal
        else:
            print("INITIATION ONLY")  # use when getting the lower level agent state dim
            # TODO: define the goal
            real_goal = self.env.env_aggregator.total_max_power

        self.lower_level_timesteps = 0
        self.meta_agent.lower_level_done = False

        temp_proportion_goal = self.goal_proportion(real_goal)
        # print('temp_proportion_goal', temp_proportion_goal)
        self.meta_agent.lower_level_state = self.turn_internal_state_to_external_state(state[5:], temp_proportion_goal)

        return self.meta_agent.lower_level_state

    # @staticmethod
    def turn_internal_state_to_external_state(self, internal_state, vector_goal):
        # add goal to the lower level agent state
        # print(' len(internal_state)', len(internal_state))
        assert len(internal_state) == self.env.observation_space.shape[0] - 5, 'state length error2'

        lower_state_from_env = list(vector_goal)

        assert len(lower_state_from_env) == int(len(internal_state) / 4), 'goal length error'

        # big_list = []
        # small_list = []
        range_list = []
        wait_list = []
        for count in range(int(len(internal_state) / 4)):
            lower_state_from_env[count] = 2 * lower_state_from_env[count] / \
                                          self.env.env.env_aggregator.evcssp_evs_objects[count].transformer_limit
            lower_state_from_env[count] += internal_state[4 * count]
            # big_list.append(lower_state[4 * count + 2])
            # small_list.append(lower_state[4 * count])
            range_list.append(internal_state[4 * count + 2] - internal_state[4 * count])
            wait_list.append(internal_state[4 * count + 3])

        lower_state_from_env += range_list
        lower_state_from_env += wait_list
        # lower_state_from_env = list(internal_state)
        # lower_state_from_env.append(float(goal))
        # lower_state_from_env = np.array(lower_state_from_env)
        return np.array(lower_state_from_env)

    def goal_proportion(self, real_goal):
        # TODO:####
        goal_feasible = min(max(float(real_goal), self.env.env.real_state[2]), self.env.env.real_state[4])
        remain_goal = goal_feasible - self.env.env.real_state[2]

        lower_state = self.env.env.real_state[5:]
        range_list = []
        for count in range(int(len(lower_state) / 4)):
            range_list.append(lower_state[4 * count + 2] - lower_state[4 * count])
        range_sum = sum(range_list)
        if range_sum > 0:
            norm_range_list = [(r_l * remain_goal) / range_sum for r_l in range_list]
        else:
            norm_range_list = [0 for _ in range_list]

        return norm_range_list

    def step(self, action):
        # print('inter step method in lower level agent env wrapper')
        # print('lower step number', self.lower_count)
        # self.lower_count += 1

        # import random
        # if random.random() < 0.008:
        #     print("Rolling intrinsic rewards {}".format(np.mean(self.track_intrinsic_rewards[-100:])))

        self.meta_agent.step_lower_level_states.append(self.meta_agent.lower_level_state)
        self.meta_agent.step_lower_level_action_seen.append(action)

        self.lower_level_timesteps += 1

        self.env.env.simulate = False
        next_state, extrinsic_reward, done, _ = self.env.step(action)

        assert len(next_state) == self.env.observation_space.shape[0], 'state length error5'
        self.update_rewards(extrinsic_reward, self.meta_agent.lower_level_state, action)
        self.update_goal()
        self.update_state_and_next_state(next_state)
        self.update_done(done)

        return self.meta_agent.lower_level_next_state, self.meta_agent.lower_level_reward, self.meta_agent.lower_level_done, _

    def update_rewards(self, extrinsic_reward, lower_state, action):
        self.meta_agent.higher_level_reward = self.calculate_high_level_reward()
        self.meta_agent.lower_level_reward = 0.1 * extrinsic_reward + self.calculate_intrinsic_reward(lower_state,
                                                                                                      action)
        # print('self.meta_agent.higher_level_reward', self.meta_agent.higher_level_reward)
        # print('self.meta_agent.lower_level_reward', self.meta_agent.lower_level_reward)

    def update_goal(self):
        self.meta_agent.goal = EVCSSPHRL_v2.goal_transition(self.meta_agent.goal)

    def update_state_and_next_state(self, next_state):
        # TODO
        assert len(next_state) == self.env.observation_space.shape[0], 'state length error1'
        # TODO:更新上层的状态
        self.meta_agent.higher_level_next_state = next_state[[0, 1, 2, 4]]
        # self.meta_agent.higher_level_next_state = next_state[:5]

        proportional_goal = self.goal_proportion(self.meta_agent.real_goal)

        # TODO:更新下层的状态
        self.meta_agent.lower_level_next_state = self.turn_internal_state_to_external_state(next_state[5:],
                                                                                            proportional_goal)
        self.meta_agent.higher_level_state = self.meta_agent.higher_level_next_state
        self.meta_agent.lower_level_state = self.meta_agent.lower_level_next_state

    def update_done(self, done):
        self.meta_agent.higher_level_done = done
        # set done flag
        # print('self.lower_level_timesteps', self.lower_level_timesteps)
        self.meta_agent.lower_level_done = done or self.lower_level_timesteps >= self.max_sub_policy_timesteps
        # print('higher done', self.meta_agent.higher_level_done)
        # print('lower done', self.meta_agent.lower_level_done)

    def calculate_intrinsic_reward(self, lower_state, action):
        """Calculates the intrinsic reward for the agent according to whether it has made progress towards the goal
        or not since the last timestep"""
        count_temp = int(len(lower_state) / 3)
        action_goal = lower_state[:count_temp]
        assert len(action_goal) == len(action)
        error = action_goal - action
        intrinsic_reward = -(np.dot(error, error)) ** 0.5
        self.track_intrinsic_rewards.append(intrinsic_reward)
        return intrinsic_reward

    def calculate_high_level_reward(self):
        # expected_reward = 0
        real_goal = self.meta_agent.real_goal
        if real_goal > self.env.env.real_state[4]:
            expected_reward = -abs(real_goal - self.env.env.real_state[4]) / 50
        elif real_goal < self.env.env.real_state[2]:
            expected_reward = -abs(real_goal - self.env.env.real_state[2]) / 50
        else:
            expected_reward = -real_goal * self.env.env.state[1] / 50
        expected_reward = float(expected_reward)
        # print('expected_reward', expected_reward)
        return expected_reward


if __name__ == '__main__':
    pass