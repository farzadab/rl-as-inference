'''
SVI tests for RL
'''
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil
import time
import json
import os

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from pyro.distributions.testing.fakes import NonreparameterizedNormal

from utils.args import str2bool
from envs.pointmass import PointMass
from .tracegraph_elbo import TraceGraph_ELBO
from .distributions import UnnormExpBernoulli, InfiniteUniform
from .svi_simple import save_everything, plot_elbo, get_args


def init_state(env):
    env.reset()
    mx_pos = th.ones(1)
    env.state[0:1] = p = pyro.sample("s", dist.Uniform(-1 * mx_pos, mx_pos))[0]
    env.state[1:2] = g = pyro.sample("g", dist.Uniform(-1 * mx_pos, mx_pos))[0]
    env.state[2:3] = 0
    return env.state

def compute_reward(a, p, g):
    return -1 * (a - (th.FloatTensor(g) - th.FloatTensor(p))).pow(2).sum()

def rl_model(env, _policy, _critic, args):
    crew = 0
    s = init_state(env)
    for i in range(args.ep_len):
        a = pyro.sample("a_%d" % i, InfiniteUniform(1))
        # reward is the distance to the correct direction
        s, _, _, _ = env.step(a.detach())
        r = compute_reward(a, s[0:1], s[1:2])
        O_dist = UnnormExpBernoulli(th.FloatTensor([r / args.ep_len]))
        pyro.sample("O_%d" % i, O_dist, obs=1)
        crew += r

    return th.cat([a.detach(), th.FloatTensor([crew])])
    

def rl_guide(env, policy, critic, args):
    pyro.module('policy', policy)
    if critic is not None:
        pyro.module('critic', critic)
    
    crew = 0
    s = init_state(env)

    for i in range(args.ep_len):
        a = pyro.sample(
            "a_%d" % i,
            NonreparameterizedNormal(
                policy(th.FloatTensor(s)),
                args.policy_stdev
            ),
            infer={'baseline': {
                'use_decaying_avg_baseline': args.use_decay_baseline and not args.use_nn_baseline,
                'nn_baseline': critic,
                'nn_baseline_input': th.cat([th.FloatTensor(s), th.FloatTensor([i])]).detach()
            }},
        )
        s, _, _, _ = env.step(a.detach())
        r = compute_reward(a, s[0:1], s[1:2])
        crew += r

    return th.cat([a.detach(), th.FloatTensor([crew])])



def create_policy_net(nb_layers, layer_size):
    return nn.Sequential(*
        [nn.Linear(3, layer_size), nn.LeakyReLU(),]
        + sum([
            [nn.Linear(layer_size, layer_size), nn.LeakyReLU()]
            for _ in range(nb_layers)
        ], [])
        + [nn.Linear(layer_size, 1)]
    )

def create_critic_net(nb_layers, layer_size):
    return nn.Sequential(*
        [nn.Linear(4, layer_size), nn.LeakyReLU(),]
        + sum([
            [nn.Linear(layer_size, layer_size), nn.LeakyReLU()]
            for _ in range(nb_layers)
        ], [])
        + [nn.Linear(layer_size, 2)]
    )

def train(args):
    env = PointMass()  # we don't care about the reward since we're not using it
    policy = create_policy_net(nb_layers=args.nb_layers, layer_size=args.layer_size)
    critic = None
    if args.use_nn_baseline:
        critic = create_critic_net(nb_layers=args.critic_nb_layers, layer_size=args.critic_layer_size)

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model=rl_model,
                         guide=rl_guide,
                         optim=pyro.optim.Adam({"lr": args.lr}),
                         loss=TraceGraph_ELBO(num_particles=args.nb_particles))

    losses = []
    for t in range(args.nb_steps):
        # step() takes a single gradient step and returns an estimate of the loss
        losses.append(svi.step(env, policy, critic, args))
        print('\rStep %d' % (t+1), end='')

    plot_elbo(losses)
    save_everything(args, policy, critic, losses)


def load(args):
    env = PointMass()  # we don't care about the reward since we're not using it
    policy = create_policy_net(nb_layers=args.nb_layers, layer_size=args.layer_size)
    policy.load_state_dict(
        th.load(os.path.join(args.load_path, 'policy.pt'))
    )
    critic = None  # TODO: load critic from file if exists
    rewards = []
    for i in range(args.nb_steps):
        rewards.append(rl_guide(env, policy, critic, args)[-1].item())
    print(sum(rewards) / args.ep_len / args.nb_steps)
    if args.debug:
        import ipdb
        ipdb.set_trace()


def main():
    args = get_args()
    if args.load_path:
        load(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
