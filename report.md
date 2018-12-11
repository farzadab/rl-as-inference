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

Carry over (Change and/or params for Future runs):
  ✔ num_particles=10  and lr=0.003


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

Carry over:
  ✔ num_particles=100 and lr=0.001
  ✔ using large networks

Experiments:
  - [1] `2018-12-08_01-49-04__svi_simple_test3`
  - [2] `2018-12-08_01-50-01__svi_simple_test3`


## Positive reward

Report:
  - Works!
    * seems like `FlexibleBernoulli` can handle positive reward, i.e. unnormalized bernoulli [3]

Remarks:
  - Don't even need to calculate the `exp` anymore so changing the name to `UnnormExpBernoulli`

Carry over:
  ✔ can use positive rewards
  ✔ no need to calculate the `exp`

Experiments:
  - [3] `2018-12-08_01-58-06__positive-reward`


## 4th test: multiple (unrelated) actions

Notes:
  - The tests up until now used Trace_ELBO not TraceGraph_ELBO, since it didn't make a difference. However, they do differ from now on. The next test addresses this issue.

Report:
  - nb_particles=100, lr=0.001, T=2 : performance is degraded, but still works [4]
    * MSE of [3] was 0.21 but it goes up to 0.51 in [4]
    * stabilizes after only 250 steps (considering T=2 this is the same as 0.5K in [1])
  - nb_particles=100, lr=0.001, T=10: performance is really bad [5]
    * MSE goes up to 1.16 in
    * doesn't stabilize after 2K steps, maybe more training is needed
  - nb_particles=100, lr=0.001, T=5 : works great (why?!) [6,9]
    * MSE of [6] is 0.15 and [9] is 0.12 (even lower than [3] ??)
    * stabilizes after 0.5K (considering T=5 this is 5 times [1] but the same in terms of optim steps)
  - nb_particles=200, lr=0.001, T=10: not yet stabilized [7]
    * MSE: 0.23, seems to get the direction right, but the values are still a bit off
  - nb_particles=100, lr=0.001, T=10: doesn't seem to stabilize (variance is high) [8]
    * MSE: 0.63

Carry over:
  ✔ __nb_particles=200__ seems to work a bit better

Experiments:
  - [4] `2018-12-08_02-29-58__multi-actions_2`
  - [5] `2018-12-08_11-44-19__multi-actions_10`
  - [6] `2018-12-09_14-10-53__multi-actions_5`
  - [7] `2018-12-09_15-56-42__multi-action_10-2`
  - [8] `2018-12-09_19-18-16__multi-action_10-3`
  - [9] `2018-12-09_14-55-56__multi-action_5-2`


## 5th test: TraceGraph (instead of Trace_ELBO)

Report:
  - TraceGraph, nb_particles=100, lr=0.001, T=10: MSE-0.21 stabilizes after 2K [10]
  - SimpleTrace

Carry over:
  ✔ TraceGraph can make it a whole lot better

Experiments:
  - [10] `2018-12-09_20-31-34__tracegraph`

## 6th test: baselines

Report:
  - decaying_avg_baseline, nb_particles=100: MSE: 0.09-0.19 and [11,12]
    * stabilizes after 0.5K (great!) and in the second run only had nb_steps=0.5K so it can get better
  - decaying_avg_baseline, nb_particles=200: MSE: **0.60** (why?) [13]
  - nn baseline (l): MSE: 0.10 with both # of particles 200 [14] and 100 [15]
  - nn baseline (m): MSE: 0.16 [16]
  - nn baseline (s): MSE: 0.34 [17]

Hyper-params:
  - 4 layers of size 16 (l)
  - 2 layers of size 8  (m)
  - 1 layers of size 4  (s)
  - nb_particles=200, T=10, lr=0.001 (unless specified)

Remark:
  - seems like __decaying_avg_baseline__ and __large enough nn_baseline__ work almost as good
  - Can we make the net larger still? Probably not worth it
  - Why did we get bad results for nb_particles=200 and decaying_avg?

Carry over:
  ✔ Let's go with __large enough NN__ since the results are more __consistent__

Experiments:
  - [11] `2018-12-09_20-41-49__baselines1_T10_graph`
  - [12] `2018-12-09_20-59-41__baselines1_T10_graph_repeat`
  - [13] `2018-12-10_00-52-38__baselines1_T10_200`
  - [14] `2018-12-10_01-52-36__baselines2_T10_200`
  - [15] `2018-12-10_01-53-13__baselines2_T10_100`
  - [16] `2018-12-10_01-51-39__baselines3_T10_200`
  - [17] `2018-12-10_01-55-51__baselines3_T10_200`


## 7th test: larger episode lengths

Report:
  - T=20: MSE: 0.15 converges after 0.6K [18]
  - T=30:
  - T=50:
  - T=100:
  - T=20, lr=0.0005: MSE: 0.10 converges after 1.1K (almost double) [19]

Experiments:
  - [18] `2018-12-10_13-32-58__T20_200_LNN`
  - [19] `2018-12-10_13-33-51__T20_200_LNN_small-LR`

# 8th test: 1D PointMass

Report:
  - T=10: MSE: 0.097 (and it actually works!)   [1]
  - T=20: MSE: 0.019 (works even better, why??) [2]

Experiments:
  - [20] `2018-12-10_14-35-15__pm1d_T10_P200`
  - [21] `2018-12-10_15-30-23__pm1d_T20_P200`

Commit: e1463d27bc959c94132db76055cfac597a6e6d4b
