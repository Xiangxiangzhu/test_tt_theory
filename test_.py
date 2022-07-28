from test_models import GATNNQ, My_MLP
from test_theory_plot import get_standard_plot
from torch.utils.tensorboard import SummaryWriter
import pickle
import torch.nn.functional as F
from env_test import EnvTest
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

# use_target = True
# use_baseline = False
use_baseline = True
episode_round = 20 * 5

seed = 0
torch.manual_seed(0)
np.random.seed(0)
random.seed(seed)


def list_to_tensor(list_):
    if not isinstance(list_, list):
        list_ = [list_]
    temp = torch.tensor(list_)
    temp = temp.view(1, -1, 1)
    temp = torch.tensor(temp, dtype=torch.float32)
    return temp


def get_action(s):
    # print("aaa ", my_model(list_to_tensor(s), list_to_tensor(0)))
    # print("bbb ", my_model(list_to_tensor(s), list_to_tensor(1)))
    if random.random() > 0.9:
        action = 0 if (my_model(list_to_tensor(s), list_to_tensor(0)) > my_model(list_to_tensor(s),
                                                                                 list_to_tensor(1))) else 1
    else:
        action = 0 if (my_model(list_to_tensor(s), list_to_tensor(0)) < my_model(list_to_tensor(s),
                                                                                 list_to_tensor(1))) else 1
    return action
    # return 1


def get_best_action(s):
    # print("aaa ", my_model(list_to_tensor(s), list_to_tensor(0)))
    # print("bbb ", my_model(list_to_tensor(s), list_to_tensor(1)))
    action = 0 if (
            my_model(list_to_tensor(s), list_to_tensor(0)) < my_model(list_to_tensor(s), list_to_tensor(1))) else 1
    return action
    # return 1


if __name__ == '__main__':
    # critic_local_mlp = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
    #                                            key_to_use="Critic")

    ###
    writer_name = "test_writer" + str(use_baseline) + str(episode_round)
    test_writer = SummaryWriter(comment=writer_name, filename_suffix="_tzm")
    ###

    record_y_ = []
    env = EnvTest()
    is_done = False
    reward_sum = 0
    while not is_done:
        print("state is ", env.state)
        s = env.state
        n_s, re, d = env.step(0)
        print("reward", re)
        print("done info", d)
        print("count number is ", env.count)
        is_done = d
        reward_sum += re

    print("total reward is ", reward_sum)

    ############################################################
    env.reset()

    ##### models ####################################################################################
    state_size = 1
    action_size = 1
    adj = torch.FloatTensor(np.ones((state_size, state_size))).unsqueeze(0)
    critic_local_gat = GATNNQ(input_state_dim=state_size, input_action_dim=action_size, output_dim=1,
                              mask_dim=state_size, emb_dim=32, window_size=1,
                              nheads=2, adj_base=adj)

    mlp_model = My_MLP(input_dim=state_size + action_size, output_dim=1)

    #############################################################################################################

    if use_baseline:
        my_model = mlp_model
    else:
        my_model = critic_local_gat

    optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-3, eps=1e-4)
    # optimizer = torch.optim.Adam(my_model.parameters(), lr=5e-2, eps=1e-4)

    for ep_id in range(episode_round):
        print("ep_id is ", ep_id)
        env.reset()
        for step_id in range(env.state_number):
            s = env.state
            # action = 0 if (my_model(list_to_tensor(s), list_to_tensor(0)) > my_model(list_to_tensor(s), list_to_tensor(1))) else 1
            action = get_action(s)
            s_, reward, done_info = env.step(action)
            if (1 > 0):
                q_next = my_model(list_to_tensor(s_), list_to_tensor(get_best_action(s_)))
                next_q_value = reward + (1 - done_info) * q_next

                loss = F.mse_loss(next_q_value, my_model(list_to_tensor(s), list_to_tensor(action)))

                optimizer.zero_grad()  # reset gradients to 0
                loss.backward()  # this calculates the gradients

                optimizer.step()  # this applies the gradients
            if done_info:
                break

        # if ((ep_id % 1) == 0) and episode_round >= 9000:
        for name, param in my_model.named_parameters():
            test_writer.add_histogram(name, param.clone().cpu().data.numpy(), ep_id)
            test_writer.add_histogram(name + "/grad", param.grad.clone().cpu().data.numpy(), ep_id)
            pass
        if ((ep_id % 1) == 0) and ep_id >= episode_round - 20:
        # if (ep_id % 1) == 0:
            # print("show result")
            x_ = [i + 1. for i in range(env.state_number)]
            y_ = [float(my_model(list_to_tensor(x), list_to_tensor(get_best_action(x))).detach().numpy()[0][0]) for x in
                  x_]
            record_y_.append(y_)

            # plt.plot(x_, y_)
            # plt.show()

    get_standard_plot(record_y_)

    plt.show()

    file_name = str(episode_round) + str(use_baseline) + '_save.pkl'
    with open(file_name, 'wb') as fp:
        pickle.dump(record_y_, fp)
