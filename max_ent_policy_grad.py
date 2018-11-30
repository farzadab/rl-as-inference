'Playing around with the PointMass environment'

# gym environment
from envs.pointmass import PointMass
import time

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

# pyro stuff for prob-proj
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# to fix pythons garbage
from copy import deepcopy

class simulator():

    def __init__(self, steps=500, policy = lambda state : np.random.rand(2) * np.ones(2), use_cuda=False):
        # length of trajectory
        self.steps = steps
        # policy given - function of the current state
        self.policy = policy
        # environment
        self.env = PointMass()
        # position of goal
        #self.goal =

    def render_trajectory(self):

        env = PointMass()
        state = env.reset()
        env.render()
        for i in range(100):
            next_state, reward, done, extra = env.step(self.policy(torch.FloatTensor(state)))
            state = next_state
            print('Reward achieved:', reward)
            env.render()
            time.sleep(env.dt * 0.5)

    def simulate_trajectory(self):

        # initialize environment
        env = PointMass()
        state = self.env.reset()

        # initialize trajectory
        trajectory = []

        # main for loop
        for i in range(self.steps):
            action = self.policy(torch.FloatTensor(state))
            prevstate = state
            state, reward, done, extra = self.env.step(action)
            # need to update draw from action
            trajectory.append([state, prevstate, action, reward])

        return trajectory

class NeuralNet(nn.Module):

    def __init__(self, state_size = 6, hidden_size = 32, variable_dimention = 2):
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

        # output parameters for a normal_dist mean + cov
        self.fc2 = nn.Linear(hidden_size, variable_dimention + variable_dimention**2)
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
        variable_dimention = self.parameterization.get_variable_dimention()

        # get mean parameters
        mean = parameters[:variable_dimention]
        # get matrix used to create covariance matrix
        dec = parameters[variable_dimention:].reshape((variable_dimention, variable_dimention))
        #dec += torch.FloatTensor(np.random.rand(variable_dimention,variable_dimention))

        # create covariance matrix
        cov = mm(dec, dec.t())

        # add a bit to the diagnol to remove degeneracy
        diagnol = 0.0001*torch.ones(variable_dimention)
        cov = torch.diag(diagnol) + cov

        # create MultivariateNormal data type
        mvn = MultivariateNormal(mean, cov)

        # return
        return mvn

    def get_parameters(self, x):

        # parameter output from nueral net
        parameters = self.parameterization(x)
        variable_dimention = self.parameterization.get_variable_dimention()

        # get mean parameters
        mean = parameters[:variable_dimention]
        # get matrix used to create covariance matrix
        dec = parameters[variable_dimention:].reshape((variable_dimention, variable_dimention))
        #dec += torch.FloatTensor(np.random.rand(variable_dimention,variable_dimention))

        # create covariance matrix
        cov = mm(dec, dec.t())

        # add a bit to the diagnol to remove degeneracy
        diagnol = 0.001*torch.ones(variable_dimention)
        cov = torch.diag(diagnol) + cov

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

    #def get_gradient_mvn():


def my_gradient_function(net, trajectories):

    # becuase we are drawing state-action pairs we should be able to use sample mean
    sample_mean = 0

    # initialize distribution
    MVN = Param_MultivariateNormal(net)

    # what is the length of trajectory
    trajectory_len = len(trajectories[0])
    trajectories_len = len(trajectories)

    # iterate through time step
    for t in range(trajectory_len):

        # average for a time t
        sample_mean = 0.0

        # iterate through all trajectories
        for i in range(trajectories_len):

            # get r(s_t,a_t), s_t, and a_t ~ q_theta(a_t|s_t)
            state = torch.FloatTensor(trajectories[i][t][0])
            prevstate = torch.FloatTensor(trajectories[i][t][1])
            action = trajectories[i][t][2]
            reward = trajectories[i][t][3]

            # get log probability of observation
            log_q = MVN.evaluate_log_pdf(action, prevstate)

            # add all rewards ahead of it
            roll_out = 0

            for t_prime in range(t,trajectory_len):

                # get state info
                state_t_prime = torch.FloatTensor(trajectories[i][t_prime][0])
                prevstate_t_prime = torch.FloatTensor(trajectories[i][t_prime][1])
                action_t_prime = trajectories[i][t_prime][2]
                reward_t_prime = trajectories[i][t_prime][3]

                # add to roll out last term is constant.
                roll_out += reward_t_prime - MVN.evaluate_log_pdf(action_t_prime, prevstate_t_prime).detach() - 1

            # add the negative log pdf of event under policy distribution
            typed_roll_out = torch.autograd.Variable(roll_out, requires_grad=False)
            typed_log_q = torch.autograd.Variable(roll_out, requires_grad=True)

            # add sample to expectation
            sample_mean += typed_log_q*typed_roll_out

        # normalize then add to outer expection for a given time t
        sample_mean += sample_mean/trajectories_len

    # we want the negative of this becuase the optimization method minimizes
    return sample_mean/trajectory_len

def my_objective_function(net, trajectories):

    # becuase we are drawing state-action pairs we should be able to use sample mean
    sample_mean = 0

    # initialize distribution
    MVN = Param_MultivariateNormal(net)

    # what is the length of trajectory
    trajectory_len = len(trajectories[0])
    trajectories_len = len(trajectories)

    # iterate through time step
    for t in range(trajectory_len):

        # average for a time t
        sample_mean = 0.0

        # iterate through all trajectories
        for i in range(trajectories_len):

            # get r(s_t,a_t), s_t, and a_t ~ q_theta(a_t|s_t)
            state = torch.FloatTensor(trajectories[i][t][0])
            prevstate = torch.FloatTensor(trajectories[i][t][1])
            action = trajectories[i][t][2]
            reward = trajectories[i][t][3]

            # get log probability of observation
            log_q = MVN.evaluate_log_pdf(action, prevstate)


            # add to roll out last term is constant.
            sample_mean += reward - MVN.evaluate_log_pdf(action, prevstate)

        # normalize then add to outer expection for a given time t
        sample_mean += sample_mean/trajectories_len

    # we want the negative of this becuase the optimization method minimizes
    return sample_mean/trajectory_len


def train_network(epochs, trajectories_per_epoch, trajectory_length):

    # initialize nueral net outputting to normal-pdf
    net = NeuralNet()

    # set optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    #optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

    # iterate for set number of epochs
    for epoch in range(epochs):

        # set running loss
        expected_loss = 0.0

        # initialize distribution
        MVN = Param_MultivariateNormal(net)

        # set policy
        policy = lambda s_t: MVN.sample(torch.tensor(s_t))

        # initialize simulation
        sim = simulator(trajectory_length, policy)

        # generate trajectories
        trajectories = [sim.simulate_trajectory() for i in range(trajectories_per_epoch)]

        # approximate expectation
        expected_loss = my_gradient_function(net, trajectories)

        # see how the objective function is improving
        average_reward = my_objective_function(net, trajectories)

        # zero the parameter gradients
        optimizer.zero_grad()

        # backprop through computation graph
        expected_loss.backward()

        # step optimizer
        optimizer.step()

        # print loss values
        print("epoch: " + str(epoch))
        print("current loss: "  + str(average_reward))
        print("current loss gradient: "  + str(expected_loss))
        # except:
        #     print("numerical issues: (probably) \n Lapack Error in potrf : the leading minor of order 2 is not positive definite at /Users/soumith/code/builder/wheel/pytorch-src/aten/src/TH/generic/THTensorLapack.cpp:626")

    print('Finished Training!')
    return net


def main():
    # pick the number of epochs / trajectories to average over ect.
    epochs = 125
    trajectories_per_epoch = 10
    trajectory_length = 100

    # train the network
    trained_net = train_network(epochs, trajectories_per_epoch, trajectory_length)

    # create parameterized normal distribution
    MVN = Param_MultivariateNormal(trained_net)

    # create policy distribution
    policy = lambda s_t: MVN.sample(torch.tensor(s_t))

    # set up simulation
    sim = simulator(100, policy)
    # see what parameters look like.
    #print(MVN.get_parameters())

    # create a simulation or 10
    for i in range(10):
        sim.render_trajectory()

if __name__ == '__main__':
    main()
