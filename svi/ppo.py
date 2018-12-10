# from pyro.infer.tracegraph_elbo import _compute_downstream_costs
from svi.trace_elbo import _compute_log_r
from pyro.infer.tracegraph_elbo import _get_baseline_options
import torch as th
import pyro
from pyro.infer.util import (detach_iterable, torch_backward)
from pyro.inder.elbo import ELBO

epsilon = 0.2

def _baseline_cost(node, downstream_cost, guide_site):
    basline = 0.0
    (nn_baseline, nn_baseline_input, use_decaying_avg_baseline, baseline_beta,
        baseline_value) = _get_baseline_options(guide_site)
    use_nn_baseline = nn_baseline is not None
    use_baseline_value = baseline_value is not None
    # assert(not (use_nn_baseline and use_baseline_value)), \
    #     "cannot use baseline_value and nn_baseline simultaneously"
    
    if use_decaying_avg_baseline:
        dc_shape = downstream_cost.shape
        param_name = "__baseline_avg_downstream_cost_" + node
        with th.no_grad():
            avg_downstream_cost_old = pyro.param(param_name,
                                                 guide_site['value'].new_zeros(dc_shape))
            avg_downstream_cost_new = (1 - baseline_beta) * downstream_cost + \
                baseline_beta * avg_downstream_cost_old
        pyro.get_param_store()[param_name] = avg_downstream_cost_new
        baseline += avg_downstream_cost_old

    if use_nn_baseline:
        baseline += nn_baseline(detach_iterable(nn_baseline_input))
    elif use_baseline_value:
        baseline += baseline_value

    if use_nn_baseline or use_baseline_value:
        baseline_loss += th.pow(downstream_cost.detach() - baseline, 2.0).sum()

    if use_nn_baseline or use_decaying_avg_baseline or use_baseline_value:
        downstream_cost = downstream_cost - baseline

    return downstream_cost, baseline_loss
    

def PPO_TraceGraph_loss(model_trace, guide_trace, pre_guide_trace, weight):
    surrogate_loss = 0
    baseline_loss = 0
    downstream_costs = _compute_log_r(model_trace, pre_guide_trace)

    for name, guide_site in reverse(list(pre_guide_trace.nodes.items())):
        score_func_term = guide_trace.nodes[name]["score_parts"].score_function /
                            guide_site["score_parts"].score_function

        downstream_cost = downstream_costs[name]
        downstream_cost, b_loss = _baseline_cost(name, downstream_cost, guide_site)
        baseline_loss += b_loss
        surrogate_loss += (th.clamp(score_func_term, min=1-epsilon, max=1+epsilon) *
                            downstream_cost.detach()).sum()

    trainable_params = any(site["type"] == "param"
                               for trace in (model_trace, guide_trace)
                               for site in trace.nodes.values())
    if trainable_params:
        loss = weight * (-surrogate_loss + baseline_loss)
        torch_backward(loss, retain_graph=True)
    
    return loss

def PPO_loss(model, guide, pre_guide_trace, num_particles, env):
    weight = 1./num_particles
    loss = 0.0
    for model_trace, guide_trace in ELBO._get_traces(model, guide, env):
        loss += PPO_TraceGraph_loss(model_trace, guide_trace, pre_guide_trace, weight)
        pre_guide_trace = guide_trace

    return loss
