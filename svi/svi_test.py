import numpy as np
import matplotlib.pyplot as plt
import time

import torch as th

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from envs.pointmass import PointMass


def rl_model(env):
    # run simulation with random actions from the prior
    crew = 0
    env.reset()
    for i in range(50):
        a = pyro.sample("a_%d" % i, dist.Uniform(-1 * th.ones(2), th.ones(2)))
        _, r, _, _ = env.step(a.reshape(-1).detach().numpy())  # ignoring the `done` signal ...
        crew += r
        O_dist = dist.Bernoulli(th.FloatTensor([r]).exp())
        pyro.sample("O_%d" % i, O_dist, obs=1)
    
    # TODO figure out: what should the return value be? I just used cumulative reward for now
    return th.FloatTensor([crew])
    

def rl_linear_guide(env):
    # Define net params, just a linear model right now:
    W = pyro.param("W", th.randn((6, 2)))
    b = pyro.param("b", th.randn((1, 2)))

    # TODO: In section 3.2 he mentions that the distribution of transitions in q must be fixed to p
    # so I think it means we should run the simulation inside the guide too (I don't know how to do it otherwise)

    # run simulation with "guided" actions
    crew = 0
    s = env.reset()
    for i in range(50):
        a = pyro.sample("a_%d" % i, dist.Normal(th.FloatTensor(s).detach().reshape(1,-1).mm(W) + b, .2))
        s, r, _, _ = env.step(a.detach().reshape(-1).numpy())  # ignoring the `done` signal ...
        crew += r
    return th.FloatTensor([crew])



pyro.clear_param_store()
svi = pyro.infer.SVI(model=rl_model,
                     guide=rl_linear_guide,
                     optim=pyro.optim.Adam({"lr": 0.001}),
                     loss=pyro.infer.Trace_ELBO())

env = PointMass(reward_style='distsq')   # the 'distsq' reward is always negative

# posterior = pyro.infer.Importance(rl_model, num_samples=10)
# marginal = pyro.infer.EmpiricalMarginal(posterior.run(env))
# print(marginal())

losses = []
for t in range(10):
    losses.append(svi.step(env))
# The method step() takes a single gradient step and returns an estimate of the loss

print("losses:", losses)

p, = plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
p.get_figure().canvas.draw()
p.get_figure().canvas.flush_events()
time.sleep(100)