'Playing around with the PointMass environment'

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

torch.manual_seed(7)
np.random.seed(7)
random.seed(7)

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
        state, reward, done, extra = self.env.step(action)

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

class NeuralNet(nn.Module):

    def __init__(self, state_size = 6, hidden_size = 8, variable_dimention = 2):
        super(NeuralNet, self).__init__()
        # set dimention for the random variables
        self.variable_dimention = variable_dimention

        # first layer
        self.fc1 = nn.Linear(state_size, hidden_size)
        # nonlinear activation functions
        self.relu1 = nn.ReLU()

        # inner layer 1
        self.fcinner1 = nn.Linear(hidden_size, hidden_size)
        # nonlinear activation functions
        self.relu2 = nn.ReLU()

        # output parameters for a normal_dist mean + diagnol cov
        self.fc2 = nn.Linear(hidden_size, 2*variable_dimention)
        # trajectory observation
        self.observation = None

    def forward(self, x):
        # first layer
        out = self.fc1(x)
        # nonlinear activation functions
        out = self.relu1(out)
        # inner layer 1
        out = self.fcinner1(out)
        # nonlinear activation functions
        out = self.relu2(out)
        # output parameters for a normal_dist
        out = self.fc2(out)
        # return
        return out

    def get_variable_dimention(self):
        return self.variable_dimention

class Param_MultivariateNormal():

    def __init__(self, parameterization = NeuralNet()):
        # set dimention for the random variables
        self.parameterization = parameterization
        # trajectory observation
        self.observation = None

    def MVN_output(self, x):
        # parameter output from nueral net
        parameters = self.parameterization(x)

        # dimention of output
        variable_dimention = self.parameterization.get_variable_dimention()

        # get mean parameters
        mean = parameters[:variable_dimention]

        # if we want a factorized covariance
        cov = torch.mul(torch.diag(parameters[variable_dimention:]),torch.diag(parameters[variable_dimention:]))

        # create MultivariateNormal data type
        mvn = MultivariateNormal(mean, cov)

        # return
        return mvn

    def get_parameters(self, x):

        # parameter output from nueral net
        parameters = self.parameterization(x)

        # dimention of output
        variable_dimention = self.parameterization.get_variable_dimention()

        # get mean parameters
        mean = parameters[:variable_dimention]

        # if we want a factorized covariance
        cov = torch.mul(torch.diag(parameters[variable_dimention:]),torch.diag(parameters[variable_dimention:]))

        return mean, cov

    def evaluate_log_pdf(self, a, x):
        # get MultivariateNormal data type
        mvn = self.MVN_output(x)
        # get log-prob of action
        return mvn.log_prob(a)

    def sample(self, x):
        # get MultivariateNormal data type
        mvn = self.MVN_output(x)
        # sample from it
        return mvn.sample()

def get_mean(net, x):
    # get MultivariateNormal data type
    mvn = Param_MultivariateNormal(net)
    # get get mean and covariance
    mean, cov = mvn.get_parameters(x)
    # return mean
    return mean.detach()

def max_ent_policy_gradient(net, trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor):

    # becuase we are drawing state-action pairs we should be able to use sample mean
    sample_mean = 0

    # initialize distribution
    MVN = Param_MultivariateNormal(net)

    # get tensor size info
    trajectory_length = len(trajectories_reward_tensor[0,:])
    simulations =  len(trajectories_reward_tensor[:,0])

    # initialize tensor for log liklihood stuff
    logliklihood_tensor = torch.zeros([trajectory_length, simulations])

    # generate tensor for log liklihood stuff
    for time in range(trajectory_length):
        for simulation in range(simulations):
            # [simulation #, value, time step]
            logliklihood_tensor[time,simulation] = MVN.evaluate_log_pdf(trajectories_action_tensor[simulation,:,time], trajectories_state_tensor[simulation,:,time])

    # initialize cumulative running average for states ahead
    cumulative_rollout = torch.zeros([trajectory_length, simulations])

    # calculate cumulative running average for states ahead + subtract entropy term
    cumulative_rollout[trajectory_length-1,:] = trajectories_reward_tensor[:,trajectory_length-1] - logliklihood_tensor[trajectory_length-1,:]
    for time in range(trajectory_length-1):
        cumulative_rollout[time,:] = cumulative_rollout[time+1,:] + trajectories_reward_tensor[:,time] - logliklihood_tensor[time,:]

    # subtract baseline
    for time in range(trajectory_length):
        cumulative_rollout[time,:] = cumulative_rollout[time,:] - trajectories_reward_tensor[:,time]

    # detach cumulative reward from computation graph
    detached_cumulative_rollout = cumulative_rollout.detach()

    # initialize expectation tensor
    expectation_tensor = torch.zeros([trajectory_length])

    # calculate instance of expectation for timestep then calc sample mean
    for time in range(trajectory_length):
        expectation_tensor[time] = torch.sum(torch.mv(detached_cumulative_rollout, logliklihood_tensor[time,:]))/simulations

    # sum accross time
    sum_expectation_tensor = torch.sum(expectation_tensor)

    # return objective with rollout detached from computation graph
    return sum_expectation_tensor

def cumulative_reward(net, trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor):
    # becuase we are drawing state-action pairs we should be able to use sample mean
    sample_mean = 0

    # initialize distribution
    MVN = Param_MultivariateNormal(net)

    # get tensor size info
    trajectory_length = len(trajectories_reward_tensor[0,:])
    simulations =  len(trajectories_reward_tensor[:,0])

    # initialize tensor for log liklihood stuff
    logliklihood_tensor = torch.zeros([trajectory_length, simulations])

    # generate tensor for log liklihood stuff
    for time in range(trajectory_length):
        for simulation in range(simulations):
            # [simulation #, value, time step]
            logliklihood_tensor[time,simulation] = MVN.evaluate_log_pdf(trajectories_action_tensor[simulation,:,time], trajectories_state_tensor[simulation,:,time])

    # initialize expectation tensor
    expectation_tensor = torch.zeros([trajectory_length])

    # calculate instance of expectation for timestep then calc sample mean
    for time in range(trajectory_length):
        expectation_tensor[time] = torch.sum(trajectories_reward_tensor[:,time] - logliklihood_tensor[time,:])/simulations

    # sum accross time
    sum_expectation_tensor = torch.sum(expectation_tensor)

    # return objective with rollout detached from computation graph
    return sum_expectation_tensor

def max_ent_TRPO(net, old_net, beta, trajectory_tensor, old_trajectories_state_tensor):

    # becuase we are drawing state-action pairs we should be able to use sample mean
    sample_mean = 0

    # get our new trajectory tensors
    trajectories_state_tensor = trajectory_tensor[0]
    trajectories_action_tensor = trajectory_tensor[1]
    trajectories_reward_tensor = trajectory_tensor[2]

    # initialize distributions
    MVN = Param_MultivariateNormal(net)
    MVN_old = Param_MultivariateNormal(old_net)

    # get tensor size info
    trajectory_length = len(trajectories_reward_tensor[0,:])
    simulations =  len(trajectories_reward_tensor[:,0])

    # initialize tensor for log liklihood stuff
    logliklihood_tensor = torch.zeros([trajectory_length, simulations])

    # intitialize kl divergence
    kl_div = 0.0

    # generate tensor for log liklihood stuff
    for time in range(trajectory_length):
        for simulation in range(simulations):
            # compute logliklihood_tensor -> [simulation #, value, time step]
            new_logpdf = MVN.evaluate_log_pdf(trajectories_action_tensor[simulation,:,time], trajectories_state_tensor[simulation,:,time])
            old_logpdf = MVN_old.evaluate_log_pdf(trajectories_action_tensor[simulation,:,time], trajectories_state_tensor[simulation,:,time])
            logliklihood_tensor[time,simulation] = new_logpdf - old_logpdf

            # also calculate kl divergence while we are at it
            kl_div = kl_div + torch.exp(old_logpdf) * (old_logpdf - new_logpdf)

    # initialize cumulative running average for states ahead
    cumulative_rollout = torch.zeros([trajectory_length, simulations])

    # calculate cumulative running average for states ahead + subtract entropy term
    cumulative_rollout[trajectory_length-1,:] = trajectories_reward_tensor[:,trajectory_length-1] - logliklihood_tensor[trajectory_length-1,:]
    for time in range(trajectory_length-1):
        cumulative_rollout[time,:] = cumulative_rollout[time+1,:] + trajectories_reward_tensor[:,time] - logliklihood_tensor[time,:]

    # subtract baseline
    for time in range(trajectory_length):
        cumulative_rollout[time,:] = cumulative_rollout[time,:] - trajectories_reward_tensor[:,time]

    # detach cumulative reward from computation graph
    detached_cumulative_rollout = cumulative_rollout.detach()

    # initialize expectation tensor
    expectation_tensor = torch.zeros([trajectory_length])

    # calculate instance of expectation for timestep then calc sample mean
    for time in range(trajectory_length):
        expectation_tensor[time] = torch.sum(torch.mv(detached_cumulative_rollout, logliklihood_tensor[time,:]))/simulations

    # sum accross time
    sum_expectation_tensor = torch.sum(expectation_tensor)

    # now add the kl divergence wrt state under the two policies
    sum_expectation_tensor = sum_expectation_tensor - beta * kl_div

    # return objective with rollout detached from computation graph
    return sum_expectation_tensor


def train_network(epochs, trajectories_per_epoch, trajectory_length):

    # initialize nueral net outputting to normal-pdf
    net = NeuralNet()

    # set optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    #optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.5)
    #optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

    # initialize tensors things to make tensors generate-able
    MVN = Param_MultivariateNormal(net)
    policy = lambda s_t: MVN.sample(torch.tensor(s_t))
    sim = simulator(trajectory_length, policy)

    # initialize initialize tensors used below
    trajectory_states, trajectory_actions, trajectory_rewards = sim.simulate_trajectory()
    trajectories_state_tensor = torch.zeros([trajectories_per_epoch, len(trajectory_states[:,0]), trajectory_length])
    trajectories_action_tensor = torch.zeros([trajectories_per_epoch, len(trajectory_actions[:,0]), trajectory_length])
    trajectories_reward_tensor = torch.zeros([trajectories_per_epoch, trajectory_length])

    # initialize trajectory set so we have old_val state trajectories_state_tensor
    for trajectory_set in range(1,trajectories_per_epoch):
        # [simulation #, value, time step]
        trajectory_states, trajectory_actions, trajectory_rewards = sim.simulate_trajectory()
        trajectories_state_tensor[trajectory_set,:,:] = trajectory_states
        trajectories_action_tensor[trajectory_set,:,:] = trajectory_actions
        trajectories_reward_tensor[trajectory_set,:] = trajectory_rewards

    # set old state trajectories
    old_trajectories_state_tensor = deepcopy(trajectories_state_tensor)

    # initialize old network model
    old_net = deepcopy(net)

    # iterate for set number of epochs
    for epoch in range(epochs):

        # initialize distribution
        MVN = Param_MultivariateNormal(net)
        # set policy
        policy = lambda s_t: MVN.sample(torch.tensor(s_t))

        # initialize simulation
        sim = simulator(trajectory_length, policy)

        # approximate expectation using trajectories
        #expected_loss = max_ent_policy_gradient(net, trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor)
        beta = 1
        expected_loss = max_ent_TRPO(net, old_net, beta, [trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor], old_trajectories_state_tensor)

        # update old network model before we update current network
        old_net = deepcopy(net)

        # zero the parameter gradients
        optimizer.zero_grad()

        # backprop through computation graph
        expected_loss.backward()

        # step optimizer
        optimizer.step()

        # set old trajectories again
        old_trajectories_state_tensor = deepcopy(trajectories_state_tensor)

        # update simulate all values and update initialized tensors
        for trajectory_set in range(1,trajectories_per_epoch):
            # [simulation #, value, time step]
            trajectory_states, trajectory_actions, trajectory_rewards = sim.simulate_trajectory()
            trajectories_state_tensor[trajectory_set,:,:] = trajectory_states
            trajectories_action_tensor[trajectory_set,:,:] = trajectory_actions
            trajectories_reward_tensor[trajectory_set,:] = trajectory_rewards

        # print loss values
        print("epoch: " + str(epoch))
        print('Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()))
        # print("parameters: ")
        # print(list(net.parameters()))
        # see how the objective function is improving
        if epoch%10==0:
            average_reward = cumulative_reward(net, trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor)
            print("current cumulative reward: "  + str(average_reward))
            print("current loss gradient: "  + str(expected_loss))

    print('Finished Training!')
    return net

def main():
    # pick the number of epochs / trajectories to average over ect.
    epochs = 500
    trajectories_per_epoch = 5
    trajectory_length = 100

    # train the network
    trained_net = train_network(epochs, trajectories_per_epoch, trajectory_length)

    # create parameterized normal distribution
    MVN = Param_MultivariateNormal(trained_net)

    # create policy distribution
    # policy = lambda s_t: MVN.sample(torch.tensor(s_t))
    policy = lambda s_t: get_mean(trained_net, torch.tensor(s_t))

    # set up simulation
    sim = simulator(50, policy)

    # see what parameters look like.
    #print(MVN.get_parameters())

    input("Press Enter to see what the trajectories look like...")

    # create a simulation or 10
    for i in range(10):
        sim.render_trajectory()

if __name__ == '__main__':
    main()
