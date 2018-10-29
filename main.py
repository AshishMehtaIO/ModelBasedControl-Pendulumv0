# ==========================================
# Title:  Main function
# Author: Ashish Mehta
# Date:   25 October 2018
# ==========================================

from model_learning import Model
from planner import Planner
import gym
from time import sleep
import math
import numpy as np


def perfect_model(input_vec):
    """
    Perfect pendulum model used to calculate the next state
    :param input_vec: [costh, sinth, thdot, u]
    :return: [cos(newth), sin(newth), newthdot]
    """
    costh, sinth, thdot, u = input_vec[0], input_vec[1], input_vec[2], input_vec[3]
    th = math.atan2(sinth, costh)
    g = 10.
    m = 1.
    l = 1.
    dt = 0.05

    u = np.clip(u, -2, 2)

    newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newthdot = np.clip(newthdot, -8, 8)  # pylint: disable=E1111
    return [math.cos(newth), math.sin(newth), newthdot]


def collect_data():
    M.random_sampler(200000)
    M.temporal_noise_sampler(200000)


def train_model():
    data = M.read_from_disk('data')
    train_dataset = M.dataset_generator(data[20000:, :], 16)
    val_dataset = M.dataset_generator(data[0:20000, :], 3)
    M.train_model(4, train_dataset, val_dataset)


def plan():
    P = Planner(M.predict_using_model)

    observation = env.reset()
    env.render()

    # define start and goal states
    start_state = [math.atan2(observation[1], observation[0]), observation[2]]
    goal_state = [0.0, 0.0]

    # plan a path using A*

    path = P.find_path(start_state=start_state, goal_state=goal_state)

    action_seq = path[:, 2]
    action_seq = action_seq[:-10]
    for ind, act in enumerate(action_seq):
        ob, _, _, _ = env.step([act])
        state = [math.atan2(ob[1], ob[0]), ob[2]]
        # print('Expected node ', path_seq[ind])
        # print('Visited node ', P.state_to_node(state))
        print('\n')
        env.render()
        sleep(0.2)

    # Loop to replan from current state
    while True:
        P = Planner(M.predict_using_model)
        start_state = state
        path = P.find_path(start_state=start_state, goal_state=goal_state)
        action_seq = path[:, 2]
        action_seq = action_seq[:-10]
        for ind, act in enumerate(action_seq):
            ob, _, _, _ = env.step([act])
            state = [math.atan2(ob[1], ob[0]), ob[2]]
            # print('Expected node ', path_seq[ind])
            # print('Visited node ', P.state_to_node(state))
            print('\n')
            env.render()
            sleep(0.2)


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    M = Model(False, "./tmp/model200000.ckpt")

    # Uncomment to collect data
    # collect_data()

    # Uncomment to train model
    train_model()


    # Uncomment to plan
    # plan()