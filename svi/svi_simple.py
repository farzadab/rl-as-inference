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

from envs.pointmass import PointMass
from .tracegraph_elbo import TraceGraph_ELBO
from .distributions import UnnormExpBernoulli, InfiniteUniform


def init_state():
    mx_pos = th.ones(2)
    p = pyro.sample("s", dist.Uniform(-1 * mx_pos, mx_pos))
    g = pyro.sample("g", dist.Uniform(-1 * mx_pos, mx_pos))
    return p, g

def compute_reward(a, p, g):
    return -1 * (a - (g - p)).pow(2).sum()

def rl_model(_, args):
    crew = 0
    p, g = init_state()
    for i in range(args.ep_len):
        a = pyro.sample("a_%d" % i, InfiniteUniform(2))
        # reward is the distance to the correct direction
        r = compute_reward(a, p, g)
        O_dist = UnnormExpBernoulli(th.FloatTensor([r / args.ep_len]))
        pyro.sample("O_%d" % i, O_dist, obs=1)
        crew += r

    return th.cat([a.squeeze().detach(), th.FloatTensor([crew])])
    

def rl_guide(policy, args):
    pyro.module('policy', policy)

    crew = 0
    p, g = init_state()

    for i in range(args.ep_len):
        a = pyro.sample(
            "a_%d" % i,
            NonreparameterizedNormal(
                policy(th.cat([p, g])),
                args.policy_stdev
            ),
        )
        r = compute_reward(a, p, g)
        crew += r

    return th.cat([a.squeeze().detach(), th.FloatTensor([crew])])


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
    plt.title("negative ELBO")
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
    parser.add_argument("--manual_debug", type=bool, default=False)
    parser.add_argument("--exp_name", type=str, default=os.path.splitext(os.path.basename(__file__))[0])
    parser.add_argument("--nb_layers", type=int, default=4)
    parser.add_argument("--layer_size", type=int, default=16)
    parser.add_argument("--nb_steps", type=int, default=1000)
    parser.add_argument("--nb_particles", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--policy_stdev", type=float, default=0.5)
    parser.add_argument("--ep_len", type=int, default=10)
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
        losses.append(svi.step(policy, args))
        print('\rStep %d' % (t+1), end='')

    plot_elbo(losses)
    save_everything(args, policy, losses)


def load(args):
    policy = create_policy_net(nb_layers=args.nb_layers, layer_size=args.layer_size)
    policy.load_state_dict(
        th.load(os.path.join(args.load_path, 'policy.pt'))
    )
    if args.manual_debug:
        import ipdb
        ipdb.set_trace()
    rewards = []
    for i in range(args.nb_steps):
        rewards.append(rl_guide(policy, args)[-1].item())
    print(sum(rewards) / args.ep_len / args.nb_steps)


def main():
    args = get_args()
    if args.load_path:
        load(args)
    else:
        train(args)


if __name__ == '__main__':
    main()