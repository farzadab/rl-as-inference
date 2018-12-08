import os
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as st

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
# for CI testing
smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)

# gym environment
from envs.pointmass import PointMass
import time
import datetime

# torch stuff for nueral networks
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch import transpose, mm
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import kl
# pyro stuff for prob-proj
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# to fix pythons garbage
from copy import deepcopy
import random
from multiprocessing import Pool
#
# torch.manual_seed(7)
# np.random.seed(7)
# random.seed(7)

class simulator():

    def __init__(self, steps=500, policy = lambda state : np.random.rand(2) * np.ones(2), use_cuda=False):
        # length of trajectory
        self.steps = steps
        # policy given - function of the current state
        self.policy = policy
        # environment
        self.env = PointMass(reward_style='distsq')

    def render_trajectory(self):

        env = PointMass(reward_style='distsq')
        state = env.reset()
        env.render()
        for i in range(self.steps):

            next_state, reward, done, extra = env.step(self.policy(torch.FloatTensor(state)))
            state = next_state
            print('Reward achieved:', -reward)
            env.render()
            time.sleep(env.dt * 0.5)

    def simulate_trajectory(self):

        # initialize environment
        env = PointMass(reward_style='distsq')
        # get initial state
        state = self.env.reset()
        # get the corrosponding action for our current state
        action = self.policy(torch.FloatTensor(state))
        # get first step for reward exct.
        state, reward, done, extra = self.env.step(action[0])

        # initialize trajectory
        trajectory_states = torch.zeros([len(state), self.steps])
        trajectory_actions = torch.zeros([len(action), self.steps])
        trajectory_rewards = torch.zeros([self.steps])

        # main for loop
        for i in range(self.steps):

            # get the corrosponding action for our current state
            action = self.policy(torch.FloatTensor(state))

            # get first step for reward exct.
            state, reward, done, extra = self.env.step(action)

            # store a_t, s_t, and r_t
            trajectory_states[:, i] = torch.FloatTensor(state)
            trajectory_actions[:, i] = torch.FloatTensor(action)
            trajectory_rewards[i] = reward

        return trajectory_states, trajectory_actions, trajectory_rewards


def bayesian_policy_gradient(epochs, trajectories_per_epoch, trajectory_length):

    # initial values for regression model
    W = torch.randn((2, 6), requires_grad=True)
    b = torch.zeros((1, 2), requires_grad=True)

    # set learning rate
    alpha = .05

    # set standard deviation
    sd = .01

    # create policy function (not random)
    policy = lambda state: dist.Normal(torch.mv(W,state) + b, sd).sample()[0]

    # generate simulator with random policy to initialize stuff
    sim = simulator(trajectory_length, policy)

    # initialize initialize tensors used below
    trajectory_states, trajectory_actions, trajectory_rewards = sim.simulate_trajectory()
    trajectories_state_tensor = torch.zeros([trajectories_per_epoch, len(trajectory_states[:,0]), trajectory_length])
    trajectories_action_tensor = torch.zeros([trajectories_per_epoch, len(trajectory_actions[:,0]), trajectory_length])
    trajectories_reward_tensor = torch.zeros([trajectories_per_epoch, trajectory_length])

    # initialize tensors for grad approximation
    Z = torch.zeros((14, trajectories_per_epoch), requires_grad=False)
    Y = torch.zeros((trajectories_per_epoch), requires_grad=False)

    for epoch in range(epochs):

        # initialize Fisher information for each parameter set
        G = torch.zeros((14, 14), requires_grad=False)

        # create policy function (not random)
        policy = lambda state: dist.Normal(torch.mv(W,state) + b, sd).sample()[0]

        # generate simulator with random policy to initialize stuff
        sim = simulator(trajectory_length, policy)

        # run bayesian policy gradient to approximate gradient update
        for trajectory_set in range(1,trajectories_per_epoch):

            # simulate trjectories under W, b
            trajectory_states, trajectory_actions, trajectory_rewards = sim.simulate_trajectory()
            trajectories_state_tensor[trajectory_set,:,:] = trajectory_states
            trajectories_action_tensor[trajectory_set,:,:] = trajectory_actions
            trajectories_reward_tensor[trajectory_set,:] = trajectory_rewards

            # compute sum of log_prob of states in trajectory wrt W,b
            trajectory_log_prob = 0.0

            for t in range(trajectory_length):
                # [value, timestep]
                trajectory_log_prob = trajectory_log_prob + dist.Normal(torch.mv(W,trajectory_states[:,t]) + b, sd).log_prob(trajectory_actions[:,t])

            # also compute cumulative reward over trjectory subtract baseline
            cumulative_reward = sum(trajectory_rewards)/trajectory_length

            # compute gradient of that sum using backwards
            trajectory_log_prob.backward(trajectory_log_prob, retain_graph = True)

            print(W.grad)
            print(b.grad)

            # store values for Y
            Y[trajectory_set] = cumulative_reward

            # store values for Z
            Z[:, trajectory_set] = torch.cat([W.grad.view(12,-1), b.grad.view(2,1)], 0).view(-1)

            # update approximation of G
            G = G + torch.ger(Z[:, trajectory_set], Z[:, trajectory_set])

            # set gradient data to zero
            W.grad = W.grad*0
            b.grad = b.grad*0

        # normalize Fisher information
        G = G/(trajectories_per_epoch*trajectory_length)

        # calculate kernal matrix
        K = torch.mm(torch.transpose(Z, 0, 1), torch.mm(G.inverse(),Z))

        # calculate covariance
        sigma = 1
        C = (K + sigma*torch.eye(K.size()[0])).inverse()

        # calculate posterior mean and covariance
        ZC = torch.mm(Z, C)
        posterior_mean = torch.mv(ZC,Y)
        posterior_covariance = None # ignore for now

        # pull grad wrt W and b
        grad_W = posterior_mean[:12].view(2,6)
        grad_b = posterior_mean[12:].view(1,2)

        # now update the parameters
        W = W + alpha*grad_W/(torch.norm(grad_W, 2))
        b = b + alpha*grad_b/(torch.norm(grad_b, 2))

        print(W)
        print(b)

        # print info
        print("epoch: " + str(epoch))
        print("gradient step W: " + str(grad_W))
        print("gradient step W: " + str(grad_b))
        print("parameter values: " + str(W) + str(b))
        print('Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()))

    return W, b




""" These are global parameters to be used later """
STATE_DIMENSIONS = 6
ACTION_DIMENSIONS = 2
STATE_ACTION_DIMENSIONS = 8

def main():
    epochs = 10
    trajectories_per_epoch = 25
    trajectory_length = 500
    W, b = bayesian_policy_gradient(epochs, trajectories_per_epoch, trajectory_length)


if __name__ == '__main__':
    main()
