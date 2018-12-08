'''
3rd simple test: larger network (with state-dependent direction and MSE reward)

Report:
  - num_particles=10  and lr=0.003: settles down after 3-4K, but it's really noisy (4 layer of size 16)
  - num_particles=10  and lr=0.001: settles down after 4-5K, still noisy (avg reward seems to be higher though)
  - num_particles=100 and lr=0.003: 
  - num_particles=1   and lr=0.003: 
  - num_particles=10  and lr=0.01 : 
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

from envs.pointmass import PointMass
from .tracegraph_elbo import TraceGraph_ELBO
from .distributions import FlexibleBernoulli, InfiniteUniform


def init_state():
    mx_pos = th.ones(2)
    p = pyro.sample("s", dist.Uniform(-1 * mx_pos, mx_pos))
    g = pyro.sample("g", dist.Uniform(-1 * mx_pos, mx_pos))
    return p, g

def compute_reward(a, p, g):
    return -1 * (a - (g - p)).pow(2).sum()

def rl_model(_):
    p, g = init_state()
    a = pyro.sample("a", InfiniteUniform(2))
    # reward is the distance to the correct direction
    r = compute_reward(a, p, g)
    O_dist = FlexibleBernoulli(th.FloatTensor([r]).exp())
    pyro.sample("O", O_dist, obs=1)
    return th.cat([a.squeeze().detach(), th.FloatTensor([r])])
    

def rl_guide(policy):
    pyro.module('policy', policy)

    p, g = init_state()

    a = pyro.sample(
        "a",
        NonreparameterizedNormal(
            policy(th.cat([p, g])),
            .5
        ),
    )
    r = compute_reward(a, p, g)
    return th.cat([a.squeeze().detach(), th.FloatTensor([r])])


def create_policy_net(nb_layers, layer_size):
    return nn.Sequential(*
        [nn.Linear(4, layer_size), nn.LeakyReLU(),]
        + sum([
            [nn.Linear(layer_size, layer_size), nn.LeakyReLU()]
            for _ in range(nb_layers)
        ], [])
        + [nn.Linear(layer_size, 2)]
    )


def plot_elbo(losses):
    p, = plt.plot(losses)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    p.get_figure().canvas.draw()
    p.get_figure().canvas.flush_events()
    plt.show()


def save_everything(args, policy, losses):
    path = os.path.join('runs', '%s__%s' % (time.strftime("%Y-%m-%d_%H-%M-%S"), args.exp_name))
    os.makedirs(path, exist_ok=True)
    th.save(policy.state_dict(), os.path.join(path, 'policy.pt'))
    plt.savefig(os.path.join(path, 'elbo.png'))
    shutil.copy2(__file__, os.path.join(path, os.path.basename(__file__)))
    with open(os.path.join(path, 'args.json'), 'w') as fargs:
        json.dump(vars(args), fargs, indent=4)
    with open(os.path.join(path, 'losses.json'), 'w') as flosses:
        json.dump(losses, flosses, indent=4)
    time.sleep(1000)


def get_args():
    parser = argparse.ArgumentParser(description="Simple test3: SVI for RL")
    parser.add_argument("--load_path", type=str, default='')
    parser.add_argument("--exp_name", type=str, default=os.path.splitext(os.path.basename(__file__))[0])
    parser.add_argument("--nb_layers", type=int, default=4)
    parser.add_argument("--layer_size", type=int, default=16)
    parser.add_argument("--nb_steps", type=int, default=1000)
    parser.add_argument("--nb_particles", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()


def train(args):
    policy = create_policy_net(nb_layers=args.nb_layers, layer_size=args.layer_size)

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model=rl_model,
                         guide=rl_guide,
                         optim=pyro.optim.Adam({"lr": args.lr}),
                         loss=pyro.infer.Trace_ELBO(num_particles=args.nb_particles))

    losses = []
    for t in range(args.nb_steps):
        # step() takes a single gradient step and returns an estimate of the loss
        losses.append(svi.step(policy))
        print('\rStep %d' % (t+1), end='')

    plot_elbo(losses)
    save_everything(args, policy, losses)


def load(args):
    policy = create_policy_net(nb_layers=args.nb_layers, layer_size=args.layer_size)
    policy.load_state_dict(
        th.load(os.path.join(args.load_path, 'policy.pt'))
    )
    rewards = []
    for i in range(args.nb_steps):
        rewards.append(rl_guide(policy)[-1].item())
    print(sum(rewards) / args.nb_steps)


def main():
    args = get_args()
    if args.load_path:
        load(args)
    else:
        train(args)


if __name__ == '__main__':
    main()