from test_models import GATNNQ, My_MLP
import torch.nn.functional as F
from env_test import EnvTest
from Replay_Buffer import Replay_Buffer
import numpy as np
import matplotlib.pyplot as plt
import torch

scale = 1
clipping_norm = 5
use_target = False


def list_to_tensor(list_):
    if not isinstance(list_, list):
        list_ = [list_]
    temp = torch.tensor(list_)
    temp = temp.view(1, -1, 1)
    temp = torch.tensor(temp, dtype=torch.float32)
    return temp


def copy_model_over(from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
        to_model.data.copy_(from_model.data.clone())


def soft_update_of_target_network(local_model, target_model, tau=0.1):
    """Updates the target network in the direction of the local network but by taking a step size
    less than one so the target network's parameter values trail the local networks.
    This helps stabilise training"""

    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


if __name__ == '__main__':
    # critic_local_mlp = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
    #                                            key_to_use="Critic")

    env = EnvTest()

    # calculate dp vector
    dp = {}
    for _ in range(env.state_number):
        print("state is ", env.state)
        s = env.state
        n_s, re, d = env.step()
        dp[round(s / scale)] = re

        print("reward", re)
        print("done info", d)
        if (d):
            print(88888)
            env.reset()

    for i in range(len(dp) - 1, 0, -1):
        dp[i] += dp[i + 1]

        pass

    dp_ = []
    for id in range(env.state_number):
        dp_.append(dp[id + 1])

    env.reset()

    ##### models ####################################################################################
    state_size = 1
    adj = torch.FloatTensor(np.ones((state_size, state_size))).unsqueeze(0)
    critic_local_gat = GATNNQ(input_state_dim=state_size, input_action_dim=0, output_dim=1,
                              mask_dim=state_size, emb_dim=64, window_size=1,
                              nheads=8, adj_base=adj)
    if use_target:
        critic_local_gat_target = GATNNQ(input_state_dim=state_size, input_action_dim=0, output_dim=1,
                                         mask_dim=state_size, emb_dim=64, window_size=1,
                                         nheads=8, adj_base=adj)
        copy_model_over(critic_local_gat, critic_local_gat_target)

    mlp_model = My_MLP(input_dim=state_size, output_dim=1)

    if use_target:
        mlp_model_target = My_MLP(input_dim=state_size, output_dim=1)
        copy_model_over(mlp_model, mlp_model_target)

    #############################################################################################################
    use_baseline = True
    if use_baseline:
        my_model = mlp_model
        if use_target:
            target_model = mlp_model_target
    else:
        my_model = critic_local_gat
        if use_target:
            target_model = critic_local_gat_target

    # optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-2, eps=1e-4)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-3, eps=1e-4)
    # memory = Replay_Buffer(1000, 64, 0)
    #
    #
    # def save_experience(experience):
    #     """Saves the recent experience to the memory buffer"""
    #     memory.add_experience(*experience)

    for ep_id in range(200):
        env.reset()
        for step_id in range(env.state_number):
            s = env.state
            s_, reward, done_info = env.step()
            # # todo
            # experience = (s, -1, reward, s_, done_info)
            # save_experience(experience)

            # if len(memory) >= memory.batch_size and ep_id % 3 == 0:
            #     # aa = memory.sample()
            #     batch_s, batch_act, batch_reward, batch_s_, batch_done_info = memory.sample()
            if (1 > 0):
                batch_s = s
                batch_reward = reward
                batch_done_info = done_info
                batch_s_ = s_
                # with torch.no_grad():
                # print("reward is ", reward)


                q_next = my_model(list_to_tensor(batch_s_))
                # q_next = target_model(list_to_tensor(s_))
                next_q_value = batch_reward + (1 - batch_done_info) * q_next
                # print("next_q_value is ", next_q_value)
                loss = F.mse_loss(next_q_value, my_model(list_to_tensor(batch_s)))
                # if (s == 2):
                #     print("loss is ", loss)
                #     print("gggg is ", my_model(list_to_tensor(batch_s)))

                optimizer.zero_grad()  # reset gradients to 0
                loss.backward()  # this calculates the gradients

                # print("loss is ", loss)
                # if clipping_norm is not None:
                #     for net in [my_model]:
                #         torch.nn.utils.clip_grad_norm_(net.parameters(),
                #                                        clipping_norm)  # clip gradients to help stabilise training

                optimizer.step()  # this applies the gradients

                loss_1 = abs(next_q_value - my_model(list_to_tensor(batch_s)))

                # if (s == 2):
                #     print("loss 1 is ", loss_1)
                #     print("hhhh is ", my_model(list_to_tensor(batch_s)))

            if done_info:
                break

        if (ep_id % 10) == 0:
            print("show result")
            x_ = [i + 1 for i in range(env.state_number)]
            y_ = [my_model(list_to_tensor(x * scale)) for x in x_]
            plt.plot(x_, y_)
            plt.plot(x_, dp_)
            plt.show()
            plt.close()
