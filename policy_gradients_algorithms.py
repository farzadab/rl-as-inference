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
from pyro.contrib.autoguide import AutoDiagonalNormal
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

            # store a_t, s_t, and r_t
            trajectory_states[:, i] = torch.FloatTensor(state)
            trajectory_actions[:, i] = torch.FloatTensor(action)
            trajectory_rewards[i] = -reward

            # get first step for reward exct.
            state, reward, done, extra = self.env.step(action)

        return trajectory_states, trajectory_actions, trajectory_rewards

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear(x)
        return out

def cumulative_reward(trajectories_reward_tensor, trajectory_length, simulations):

    # initialize expectation tensor
    expectation_tensor = torch.zeros([trajectory_length])

    # calculate instance of expectation for timestep then calc sample mean
    for time in range(trajectory_length):
        expectation_tensor[time] = torch.sum(trajectories_reward_tensor[:,time])/simulations

    # sum accross time
    sum_expectation_tensor = torch.sum(expectation_tensor)

    # return expected rewards across time
    return sum_expectation_tensor

class MEPG_Loss(torch.nn.Module):

    def __init__(self, sd, alpha):

        super(MEPG_Loss, self).__init__()
        self.sd = sd
        self.alpha = alpha

    def Advantage_estimator(self, alpha, trajectory_length, simulations, logliklihood_tensor, trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor):
        """ COMPUTES ROLL OUT WITH MAX ENT REGULARIZATION """

        # initialize cumulative running average for states ahead
        cumulative_rollout = torch.zeros([trajectory_length, simulations])

        # calculate cumulative running average for states ahead + subtract entropy term
        cumulative_rollout[trajectory_length-1,:] = trajectories_reward_tensor[:,trajectory_length-1] - alpha*logliklihood_tensor[trajectory_length-1,:]
        for time in range(trajectory_length-1):
            cumulative_rollout[time,:] = cumulative_rollout[time+1,:] + trajectories_reward_tensor[:,time] - alpha*logliklihood_tensor[time,:]

        # subtract baseline
        env = PointMass(reward_style='distsq')
        env.reset()
        for time in range(trajectory_length):
            # cumulative_rollout[time,:] = cumulative_rollout[time,:] - trajectories_reward_tensor[:,time]
            mx_pos = torch.log((1 / (2 * env.max_position))* torch.ones(cumulative_rollout[time,:].size()[0]))
            cumulative_rollout[time,:] = cumulative_rollout[time,:] - mx_pos
        # detach cumulative reward from computation graph
        advantage = cumulative_rollout.detach()

        return advantage

    def forward(self, model, trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor):

        """ SET VALUES"""
        sd = self.sd
        alpha = self.alpha

        """ CALCULATE LOG LIKLIHOOD OF TRAJECTORIES """
        # get tensor size info
        trajectory_length = len(trajectories_reward_tensor[0,:])
        simulations =  len(trajectories_reward_tensor[:,0])
        # initialize tensor for log liklihood stuff
        logliklihood_tensor = torch.zeros([trajectory_length, simulations])
        # generate tensor for log liklihood stuff
        for time in range(trajectory_length):
            for simulation in range(simulations):
                # [simulation #, value, time step]
                logliklihood_tensor[time,simulation] = \
                dist.MultivariateNormal(model(trajectories_state_tensor[simulation,:,time]), sd).log_prob(trajectories_action_tensor[simulation,:,time])

        """ CALCULATE ADVANTAGE REGULARIZED BY ENTROPY """
        A_hat = self.Advantage_estimator(alpha, trajectory_length, simulations, logliklihood_tensor, \
                                         trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor)

        """ CALCULATE POLICY GRADIENT OBJECTIVE """
        # initialize expectation tensor
        expectation_tensor = torch.zeros([trajectory_length])
        # calculate instance of expectation for timestep then calc sample mean
        for time in range(trajectory_length):
            expectation_tensor[time] = torch.sum(torch.mv(A_hat, logliklihood_tensor[time,:]))/simulations
        # sum accross time
        sum_expectation_tensor = torch.sum(expectation_tensor)

        """ RETURN """
        return sum_expectation_tensor

def train_max_ent_policy_gradient(epochs, trajectories_per_epoch, trajectory_length):

    """ INITIALIZATIONS """
    # initial variables for regression model
    reg_model = LinearRegressionModel(6,2)

    # set standard deviation for normal distsq
    sd = 100*torch.eye(2)

    # set optimizer
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=1e-2)
    #optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.5)
    #optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

    # initialize tensors
    trajectory_states, trajectory_actions, trajectory_rewards = simulator(trajectory_length).simulate_trajectory()
    trajectories_state_tensor = torch.zeros([trajectories_per_epoch, len(trajectory_states[:,0]), trajectory_length])
    trajectories_action_tensor = torch.zeros([trajectories_per_epoch, len(trajectory_actions[:,0]), trajectory_length])
    trajectories_reward_tensor = torch.zeros([trajectories_per_epoch, trajectory_length])

    # cooling function to reduce stochasticity of gradient near optimal solutions
    alpha = 1
    cooling = lambda alpha: alpha/1.05

    # initialize loss function class
    loss_mod = MEPG_Loss(sd, alpha)

    # apply PG iteratively
    for epoch in range(epochs):

        """ SIMULATE TRAJECTORIES UNDER CURRENT POLICY """
        # update alpha
        alpha = cooling(alpha)

        # create policy function
        policy = lambda state: dist.MultivariateNormal(reg_model(torch.tensor(state)), sd).sample()

        # generate simulator
        sim = simulator(trajectory_length, policy)

        # data to approximate gradient with
        for trajectory_set in range(trajectories_per_epoch):
            # [simulation #, value, time step]
            trajectory_states, trajectory_actions, trajectory_rewards = sim.simulate_trajectory()
            trajectories_state_tensor[trajectory_set,:,:] = trajectory_states
            trajectories_action_tensor[trajectory_set,:,:] = trajectory_actions
            trajectories_reward_tensor[trajectory_set,:] = trajectory_rewards

        """ PERFORM OPTIMIZATION STEP ON OBJECTIVE """
        # loss objective being minimized
        loss = loss_mod(reg_model, trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor)

        # zero the parameter gradients
        optimizer.zero_grad()

        # backprop through computation graph
        loss.backward()

        # step optimizer
        optimizer.step()

        """ PRINT INFO """
        # print info
        if epoch%5==0:
            avg = cumulative_reward(trajectories_reward_tensor, trajectory_length, trajectories_per_epoch)
            print("epoch: " + str(epoch))
            print('Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()))
            print("current loss gradient: "  + str(loss))
            print("current cumulative reward: "  + str(avg))
            print(reg_model.linear.weight, reg_model.linear.bias)

    # print and return
    print("Algorithm complete!")
    return reg_model.linear.weight, reg_model.linear.bias

def train_bayesian_policy_gradient(epochs, trajectories_per_epoch, trajectory_length):

    # initial values for regression model
    W = torch.randn((2, 6), requires_grad=True)
    b = torch.zeros((1, 2), requires_grad=True)

    # set learning rate
    alpha = .01

    # set standard deviation
    sd = 1

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
        for trajectory_set in range(trajectories_per_epoch):

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
            cumulative_reward = sum(trajectory_rewards)

            # compute gradient of that sum using backwards
            trajectory_log_prob.backward(trajectory_log_prob)

            # store values for Y
            Y[trajectory_set] = cumulative_reward

            # store values for Z
            Z[:, trajectory_set] = torch.cat([W.grad.view(12,-1), b.grad.view(2,1)], 0).view(-1)

            # update approximation of G
            G = G + torch.ger(Z[:, trajectory_set], Z[:, trajectory_set])

            # set gradient data to zero
            W.grad = W.grad*0
            b.grad = b.grad*0

        # gradient descent, don't track
        with torch.no_grad():

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

            # normalize gradient so the objective doesnt explode.
            W = W + alpha*grad_W/(torch.norm(grad_W, 2))
            b = b + alpha*grad_b/(torch.norm(grad_b, 2))

        # reinitialize parameters
        W.requires_grad = True
        b.requires_grad = True

        # scale down step size (bc I dont use line search)
        # alpha = alpha / 1.01

        # print info
        print("epoch: " + str(epoch))
        print("expected return: ", cumulative_reward(trajectories_reward_tensor, trajectory_length, trajectories_per_epoch))
        print('Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()))
        print('w:', W)
        print('b:', b)
        print('w.grad:', grad_W/(torch.norm(grad_W, 2)))
        print('b.grad:', grad_b/(torch.norm(grad_b, 2)))

    return W, b

""" These are global parameters to be used later """
STATE_DIMENSIONS = 6
ACTION_DIMENSIONS = 2
STATE_ACTION_DIMENSIONS = 8

def main():

    """ INITIALIZATIONS """
    # initialization stuff
    epochs = 1000
    trajectories_per_epoch = 25
    trajectory_length = 500

    """ MAX ENTROPY POLICY GRADIENTS """
    # lets try this with max ent policy gradients
    W, b = train_max_ent_policy_gradient(epochs, trajectories_per_epoch, trajectory_length)
    print(W, b)
    # set up simulation
    policy = lambda state: dist.Normal(torch.mv(W,state) + b, 1).sample()
    sim = simulator(500, policy)

    # create a simulation or 10
    input("Press Enter to see what the trajectories look like...")
    for i in range(10):
        sim.render_trajectory()

    # lets try this with max ent TRPO


    """ MAX ENTROPY BAYESIAN / NATURAL POLICY GRADIENTS """
    # lets try this with bayesian / natural policy gradients
    W, b = train_bayesian_policy_gradient(epochs, trajectories_per_epoch, trajectory_length)
    print(W, b)
    # set up simulation
    policy = lambda state: dist.Normal(torch.mv(W,state) + b, 0.0001).sample()[0]
    sim = simulator(250, policy)

    input("Press Enter to see what the trajectories look like...")

    # create a simulation or 10
    for i in range(10):
        sim.render_trajectory()

    """ MAX ENTROPY POLICY GRADIENTS VIA BAYESIAN REGRESSION IN PYRO """
    # set prior over weights
    prior = dist.Normal(torch.mv(W,state) + b, 1)
    W, b = train_BR_policy_gradient(prior, epochs, trajectories_per_epoch, trajectory_length)
    print(W, b)
    # set up simulation
    policy = lambda state: dist.Normal(torch.mv(W,state) + b, 0.0001).sample()[0]
    sim = simulator(250, policy)

    input("Press Enter to see what the trajectories look like...")

    # create a simulation or 10
    for i in range(10):
        sim.render_trajectory()

if __name__ == '__main__':
    main()
