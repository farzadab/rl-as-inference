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
    mx_pos = th.ones(env.dim) * env.max_position
    mx_vel = th.ones(env.dim) * env.max_speed
    env.state[        0:  env.dim] = p = pyro.sample("s", dist.Uniform(-1 * mx_pos, mx_pos))
    env.state[  env.dim:2*env.dim] = g = pyro.sample("g", dist.Uniform(-1 * mx_pos, mx_pos))
    env.state[2*env.dim:3*env.dim] = v = pyro.sample("v", dist.Uniform(-1 * mx_vel, mx_vel))
    return env.state

def compute_reward(a, s, args):
    p = s[           0:  args.env_dim]
    g = s[args.env_dim:2*args.env_dim]
    return -1 * (a - (th.FloatTensor(g) - th.FloatTensor(p))).pow(2).sum()

def rl_model(env, _policy, _critic, args):
    crew = 0
    s = init_state(env)
    for i in range(args.ep_len):
        a = pyro.sample("a_%d" % i, InfiniteUniform(1))
        # reward is the distance to the correct direction
        s, r, _, _ = env.step(a.detach())
        if not args.env_reward:
            r = compute_reward(a, s, args)
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
                policy(th.FloatTensor(s) / env.max_position) * env.max_torque,
                args.policy_stdev
            ),
            infer={'baseline': {
                'use_decaying_avg_baseline': args.use_decay_baseline and not args.use_nn_baseline,
                'nn_baseline': critic,
                'nn_baseline_input': th.cat([
                    th.FloatTensor(s) / env.max_position,
                    th.FloatTensor([i]) / args.ep_len
                ]).detach()
            }},
        )
        s, r, _, _ = env.step(a.detach())
        if not args.env_reward:
            r = compute_reward(a, s, args)
        crew += r
        if args.render:
            env.render()
            time.sleep(env.dt / 2)

    return th.cat([a.detach(), th.FloatTensor([crew])])


def create_policy_net(args):
    return nn.Sequential(*
        [nn.Linear(3*args.env_dim, args.layer_size), nn.LeakyReLU(),]
        + sum([
            [nn.Linear(args.layer_size, args.layer_size), nn.LeakyReLU()]
            for _ in range(args.nb_layers)
        ], [])
        + [nn.Linear(args.layer_size, args.env_dim)]
    )

def create_critic_net(args):
    return nn.Sequential(*
        [nn.Linear(3*args.env_dim+1, args.critic_layer_size), nn.LeakyReLU(),]
        + sum([
            [nn.Linear(args.critic_layer_size, args.critic_layer_size), nn.LeakyReLU()]
            for _ in range(args.critic_nb_layers)
        ], [])
        + [nn.Linear(args.critic_layer_size, args.env_dim)]
    )

def train(args):
    env = PointMass(dim=args.env_dim, reward_style=args.env_reward or 'pot')
    policy = create_policy_net(args)
    critic = None
    if args.use_nn_baseline:
        critic = create_critic_net(args)

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
    env = PointMass(dim=args.env_dim, reward_style=args.env_reward or 'pot')
    policy = create_policy_net(args)
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

def get_args():
    parser = argparse.ArgumentParser(description="Simple test3: SVI for RL")
    parser.add_argument("--load_path", type=str, default='')
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--render", type=str2bool, default=False)
    parser.add_argument("--env_dim", type=int, default=1)
    parser.add_argument("--env_reward", type=str, default='')
    parser.add_argument("--exp_name", type=str, default=os.path.splitext(os.path.basename(__file__))[0])
    parser.add_argument("--nb_layers", type=int, default=4)
    parser.add_argument("--layer_size", type=int, default=16)
    parser.add_argument("--critic_nb_layers", type=int, default=4)
    parser.add_argument("--critic_layer_size", type=int, default=16)
    parser.add_argument("--nb_steps", type=int, default=1000)
    parser.add_argument("--nb_particles", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--policy_stdev", type=float, default=0.5)
    parser.add_argument("--ep_len", type=int, default=10)
    parser.add_argument("--use_decay_baseline", type=str2bool, default=False)
    parser.add_argument("--use_nn_baseline", type=str2bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    main()
