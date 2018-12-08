import torch as th
import pyro.distributions as dist


class FlexibleBernoulli(dist.Distribution):
    def __init__(self, p):
        self.p = p
        self.dist = th.distributions.bernoulli.Bernoulli(p)
    def sample(self):
        print("We're screwed ...")   # you should not be sampling from this distribution
        return self.dist.sample()
    def log_prob(self, x):
        if x > 0.5:
            return th.log(th.FloatTensor([self.p]))
        print("We're doubly screwed ...")  # You should only get the log_prob from this distribution
        return self.dist.log_prob(x)