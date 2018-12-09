import torch as th
import pyro.distributions as dist


class UnnormExpBernoulli(dist.Distribution):
    def __init__(self, p):
        self.p = p
        self.dist = th.distributions.bernoulli.Bernoulli(self.p.exp())
    def sample(self):
        print("We're screwed ...")   # you should not be sampling from this distribution
        return self.dist.sample()
    def log_prob(self, x):
        if x > 0.5:
            return self.p
        print("We're doubly screwed ...")  # You should only get the log_prob from this distribution
        return self.dist.log_prob(x)

class InfiniteUniform(dist.Distribution):
    def __init__(self, dim=1):
        mx = 1000 * th.ones(dim)
        self.dist = th.distributions.uniform.Uniform(-mx, mx)
    def sample(self):
        print("We're screwed ...")   # you should not be sampling from this distribution
        return self.dist.sample()
    def log_prob(self, x):
        return th.FloatTensor([0])
