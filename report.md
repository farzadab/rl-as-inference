# Simple tests:

## Simplest test: finding a fixed direction with MSE

Report:
  - it works, but it does take a bit of time
    * depending on the starting point, takes around 200-500 steps with lr=0.01 and num_particles=100
  - lr=0.001: needs a lot more than 1000 steps
  - num_particles=10 and lr=0.001: solves after ~7K (more efficient than num_particles=100)
  - num_particles=1  and lr=0.001: solves after ~20K (a lot more efficient!)
  - num_particles=1  and lr=0.01 : solves really fast but doesn't quite settle on the right solution (moves around)
  - num_particles=10 and lr=0.01 : same as above but with less variation

  - num_particles=10 and lr=0.003: solves after ~2K
Remark:
  - a fixed value of lr * steps ~= 2-20 is required almost regardless of the num_particles
    * num_particles is important, but doesn't seem to be the bottleneck for this problem


## 2nd test: state-dependent direction with MSE reward

Report:
  - num_particles=10  and lr=0.003: solves after ~2-3K
  - num_particles=100 and lr=0.003: solves after ~2-3K (really inefficient)
  - num_particles=1   and lr=0.003: solves after ~8-9K (more efficient, but noisy ELBO)
    * can't really tell from ELBO alone when/if the solution has been found
  - num_particles=10  and lr=0.001: solves after ~7-8K (in line with the lr * steps formula from previous experiments)
  - num_particles=10  and lr=0.01 : solves after ~1-2K but doesn't settle down even after 5K


## 3rd test: larger (non-linear) network (with state-dependent direction and MSE reward)

Report:
  - num_particles=10  and lr=0.003: settles down after 3-4K, but it's really noisy
  - num_particles=10  and lr=0.001: settles down after 4-5K, still noisy (avg reward seems to be higher though)
  - num_particles=100 and lr=0.001: settles down after 0.5K [1]
  - num_particles=100 and lr=0.003: settles down after 1K   [2]
  - num_particles=1   and lr=0.003: 
  - num_particles=10  and lr=0.01 : 

Hyper-params:
  - 4 layers of size 16


Experiments:
  - [1] `2018-12-08_01-49-04__svi_simple_test3`
  - [2] `2018-12-08_01-50-01__svi_simple_test3`


## Positive reward

Report:
  - Works!
    * seems like `FlexibleBernoulli` can handle positive reward, i.e. unnormalized bernoulli [1]

Experiments:
  - [1] `2018-12-08_01-58-06__positive-reward`

Remarks:
  - Don't even need to calculate the `exp` anymore so changing the name to `UnnormExpBernoulli`


## 4th test: multiple (unrelated) actions

Report:
  - nb_particles=100, lr=0.001, T=2:  [1]

Experiments:
  - [1]