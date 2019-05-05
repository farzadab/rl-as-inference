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

class BaselineModel(nn.Module):

    """ baseline make the score equal to zero (i.e. be equal to negative of sum of rewards) """
    """ takes in state and spits out approximation of negative of sum of rewards given your in that state"""
    """ use a linear classifier becuase its fast """

    def __init__(self, input_dim, output_dim):
        super(BaselineModel, self).__init__()
        # Calling Super Class's constructor
        self.hidden_layer = 64
        self.linear1 = nn.Linear(input_dim, self.hidden_layer)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_layer, output_dim)

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
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

    """ MAXIMUM ENTROPY POLICY GRADIENTS LOSS FUNCTION """

    def __init__(self, sd, alpha):

        super(MEPG_Loss, self).__init__()
        self.sd = sd
        self.alpha = alpha
        self.base_line = None

    def Baseline_approximation(self, x, y, state_size):

        # initialize function approximation
        if self.base_line == None:
            print("baseline initialized...")

            baseline = BaselineModel(state_size, 1)
            self.base_line = baseline
            epochs = 5000
            l_rate = 0.1
            # train to approximate average reward at state
            criterion = nn.MSELoss()# Mean Squared Loss

            # define optimizer and learning rate
            optimiser = torch.optim.Adam(baseline.parameters(), lr = l_rate)

            # train model using data
            for epoch in range(epochs):

                #increase the number of epochs by 1 every time
                inputs = Variable(x)
                labels = Variable(y)

                #clear grads as discussed in prev post
                optimiser.zero_grad()

                #forward to get predicted values
                outputs = torch.cat([baseline(x_i) for x_i in inputs.transpose(0,1)])
                loss = criterion(outputs, labels)

                loss.backward() # back props
                optimiser.step() # update the parameters

                if epoch%1000==0:
                    print(np.floor(100*epoch/epochs), "% percent complete")
                    print("current loss", loss)

        else:
            baseline = self.base_line
            epochs = 500
            l_rate = 0.1
            # train to approximate average reward at state
            criterion = nn.MSELoss()# Mean Squared Loss

            # define optimizer and learning rate
            optimiser = torch.optim.Adam(baseline.parameters(), lr = l_rate)

            # train model using data
            for epoch in range(epochs):

                #increase the number of epochs by 1 every time
                inputs = Variable(x)
                labels = Variable(y)

                #clear grads as discussed in prev post
                optimiser.zero_grad()

                #forward to get predicted values
                outputs = torch.cat([baseline(x_i) for x_i in inputs.transpose(0,1)])
                loss = criterion(outputs, labels)

                loss.backward() # back props
                optimiser.step() # update the parameters

        print("baseline updated... loss at ", loss)
        return baseline

    def Advantage_estimator(self, alpha, trajectory_length, simulations, logliklihood_tensor, trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor):
        """ COMPUTES ROLL OUT WITH MAX ENT REGULARIZATION """

        # reward shaping
        trajectories_reward_tensor = torch.exp(trajectories_reward_tensor)

        # initialize cumulative running average for states ahead
        cumulative_rollout = torch.zeros([trajectory_length, simulations])

        # initialize score function info
        x = torch.zeros([6, trajectory_length*simulations])
        y = torch.zeros([trajectory_length*simulations])

        # calculate cumulative running average for states ahead + subtract entropy term
        cumulative_rollout[trajectory_length-1,:] = trajectories_reward_tensor[:,trajectory_length-1] - alpha*logliklihood_tensor[trajectory_length-1,:]

        # calculate first term in the values used in baseline estimator x = [state, time-instance] y = [cumulative reward,  time-instance]
        y[(trajectory_length - 1)*simulations:trajectory_length*simulations] = trajectories_reward_tensor[:,trajectory_length-1]
        x[:, (trajectory_length - 1)*simulations:trajectory_length*simulations] = trajectories_state_tensor[:, :, trajectory_length-1].transpose(0,1)

        # primary loop
        for time in reversed(range(trajectory_length-1)):
            # cumulative reward starting from time = time
            cumulative_rollout[time,:] = cumulative_rollout[time+1,:] + trajectories_reward_tensor[:,time] - alpha*logliklihood_tensor[time,:]
            # x =  state realization , and y = score for that state
            x[:, time*simulations:(time+1)*simulations] = trajectories_state_tensor[:, :, time].transpose(0,1)
            y[time*simulations:(time+1)*simulations] = cumulative_rollout[time+1,:] + trajectories_reward_tensor[:,time]

        # train baseline function
        base_line = self.Baseline_approximation(x, y, 6)

        #calculate baseline
        current_baseline = torch.cat([base_line(x_i) for x_i in x.transpose(0,1)])

        # subtract baseline from cumulative reward
        for time in range(trajectory_length):
            cumulative_rollout[time,:] = cumulative_rollout[time,:] - current_baseline[time*simulations:(time + 1)*simulations]

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
        sum_expectation_tensor = torch.sum(expectation_tensor)/trajectory_length

        """ RETURN """
        return sum_expectation_tensor

def train_max_ent_policy_gradient(epochs, trajectories_per_epoch, trajectory_length):

    """ INITIALIZATIONS """
    # initial variables for regression model
    reg_model = LinearRegressionModel(6,2)

    # set standard deviation for normal distribution of model
    sd = 10*torch.eye(2)

    # set optimizer
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=1e-2)
    #optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.5)
    #optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

    # initialize tensors [simulations, values, time T]
    trajectory_states, trajectory_actions, trajectory_rewards = simulator(trajectory_length).simulate_trajectory()
    trajectories_state_tensor = torch.zeros([trajectories_per_epoch, len(trajectory_states[:,0]), trajectory_length])
    trajectories_action_tensor = torch.zeros([trajectories_per_epoch, len(trajectory_actions[:,0]), trajectory_length])
    trajectories_reward_tensor = torch.zeros([trajectories_per_epoch, trajectory_length])

    # cooling function to reduce stochasticity of gradient near optimal solutions
    alpha = 1
    cooling = lambda alpha: alpha/1.0

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

        print(trajectories_per_epoch, "trajectories sampled with trajectory length ", trajectory_length)

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
        if epoch%1==0:
            avg = cumulative_reward(trajectories_reward_tensor, trajectory_length, trajectories_per_epoch)
            print("epoch: " + str(epoch))
            print('Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()))
            print("current loss gradient: "  + str(loss))
            print("current cumulative reward: "  + str(avg))
            print(reg_model.linear.weight, reg_model.linear.bias)

    # print and return
    print("Algorithm complete!")
    return reg_model.linear.weight, reg_model.linear.bias

def train_natural_policy_gradient(epochs, trajectories_per_epoch, trajectory_length):

    """ INITIALIZATIONS """
    # initial variables for regression model
    reg_model = LinearRegressionModel(6,2)

    # set standard deviation for normal distsq
    sd = 1*torch.eye(2)

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

    # initialize tensors for grad approximation
    Z = torch.zeros((14, trajectories_per_epoch), requires_grad=False)
    Y = torch.zeros((trajectories_per_epoch), requires_grad=False)

    # apply PG iteratively
    for epoch in range(epochs):

        """ SIMULATE TRAJECTORIES UNDER CURRENT POLICY """
        # update alpha
        alpha = cooling(alpha)

        # create policy function
        policy = lambda state: dist.MultivariateNormal(reg_model(torch.tensor(state)), sd).sample()

        # generate simulator
        sim = simulator(trajectory_length, policy)

        # initialize Fisher information for each parameter set
        FI = torch.zeros((14, 14), requires_grad=False)

        # create temp variables W, b describing regression model
        W = deepcopy(reg_model.linear.weight)
        b = deepcopy(reg_model.linear.bias)

        # data to approximate gradient with
        for trajectory_set in range(trajectories_per_epoch):
            # [simulation #, value, time step]
            trajectory_states, trajectory_actions, trajectory_rewards = sim.simulate_trajectory()
            trajectories_state_tensor[trajectory_set,:,:] = trajectory_states
            trajectories_action_tensor[trajectory_set,:,:] = trajectory_actions
            trajectories_reward_tensor[trajectory_set,:] = trajectory_rewards

            """ APPROXIMATE FISCHER INFORMATION"""
            # compute sum of log_prob of states in trajectory wrt W,b
            trajectory_log_prob = 0.0

            for t in range(trajectory_length):
                # [value, timestep]
                trajectory_log_prob = trajectory_log_prob + dist.MultivariateNormal(torch.mv(W,trajectory_states[:,t]) + b, sd).log_prob(trajectory_actions[:,t])

            # also compute cumulative reward over trjectory subtract baseline
            cumulative_reward_tensor = sum(trajectory_rewards)

            # compute gradient of that sum using backwards
            trajectory_log_prob.backward()

            # store values for Y
            Y[trajectory_set] = cumulative_reward_tensor

            # store values for Z
            Z[:, trajectory_set] = torch.cat([W.grad.reshape(12,-1), b.grad.reshape(2,1)], 0).view(-1)

            # update approximation of G
            FI = FI + torch.ger(Z[:, trajectory_set], Z[:, trajectory_set])

            # # set gradient data to zero
            W.grad = W.grad*0
            b.grad = b.grad*0

        # normalize Fisher information
        FI = FI/(trajectories_per_epoch*trajectory_length)

        """ PERFORM OPTIMIZATION STEP ON OBJECTIVE """
        # loss objective being minimized
        loss = loss_mod(reg_model, trajectories_state_tensor, trajectories_action_tensor, trajectories_reward_tensor)

        # zero the parameter gradients
        reg_model.linear.weight.grad = None
        reg_model.linear.bias.grad = None

        # backprop through computation graph
        loss.backward()

        """ COMPUTE NATURAL GRADIENTS """
        # get gradient
        g = torch.cat((reg_model.linear.weight.grad.view(12,-1), reg_model.linear.bias.grad.view(2,1)), 0)
        # solve system of equations to get inv_FI times grad
        FI_g = torch.gesv(g, FI)[0]
        # get natural gradient
        natural_grad = (2/(torch.dot(g.view(-1), FI_g.view(-1))))*FI_g.view(-1)

        """ UPDATE THE GRADIENTS OF THE PARAMETERS THEN STEP"""
        # update weight and bias gradients
        reg_model.linear.weight.grad = natural_grad[:12].reshape(2,6)
        reg_model.linear.bias.grad = natural_grad[12:].reshape(-1)

        # step optimizer
        optimizer.step()

        """ PRINT INFO """
        # print info
        if epoch%1==0:
            avg = cumulative_reward(trajectories_reward_tensor, trajectory_length, trajectories_per_epoch)
            print("epoch: " + str(epoch))
            print('Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()))
            print("current loss gradient: "  + str(loss))
            print("current cumulative reward: "  + str(avg))
            print(reg_model.linear.weight, reg_model.linear.bias)

    # print and return
    print("Algorithm complete!")
    return reg_model.linear.weight, reg_model.linear.bias

""" These are global parameters to be used later """
STATE_DIMENSIONS = 6
ACTION_DIMENSIONS = 2
STATE_ACTION_DIMENSIONS = 8

def main():

    """ INITIALIZATIONS """
    # initialization stuff
    epochs = 10
    trajectories_per_epoch = 5
    trajectory_length = 100

    """ MAX ENTROPY POLICY GRADIENTS WITH BASELINE APPROXIMATION """
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

    """ MAX ENTROPY NATURAL POLICY GRADIENTS """
    # lets try this with bayesian / natural policy gradients
    W, b = train_natural_policy_gradient(epochs, trajectories_per_epoch, trajectory_length)
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
