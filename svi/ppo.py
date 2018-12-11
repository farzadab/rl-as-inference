# from pyro.infer.tracegraph_elbo import _compute_downstream_costs
import torch as th
import pyro
import copy
import pyro.poutine as pp
from pyro.infer.util import (detach_iterable, torch_backward, zero_grads)
from pyro.infer.elbo import ELBO
from .trace_elbo import _compute_log_r
from pyro.infer.tracegraph_elbo import _get_baseline_options
from pyro.distributions.util import is_identically_zero

epsilon = 0.2

def _baseline_cost(node, downstream_cost, guide_site):
    baseline_loss = 0.0
    baseline = 0.0
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
    

def PPO_TraceGraph_loss(model_trace, guide_trace, old_policy_log_prob, epsilon=0.2):
    surrogate_loss = 0
    baseline_loss = 0
    downstream_costs = _compute_log_r(model_trace, guide_trace)
    non_reparam = set(guide_trace.nonreparam_stochastic_nodes)

    for name, guide_site in guide_trace.nodes.items():
        if guide_site["type"] == "sample" and (name in non_reparam):
            score_func_term = guide_site["score_parts"].score_function
            if not is_identically_zero(old_policy_log_prob[name]):
                score_func_term = score_func_term - old_policy_log_prob[name].detach()

            downstream_cost, b_loss = _baseline_cost(name, downstream_costs[name], guide_site)
            
            if not is_identically_zero(score_func_term):
                surrogate_loss += ((th.clamp(th.Tensor(score_func_term), min=1-epsilon, max=1+epsilon)) * \
                                    downstream_cost.detach()).sum()
            baseline_loss += b_loss

    trainable_params = any(site["type"] == "param"
                               for trace in (model_trace, guide_trace)
                               for site in trace.nodes.values())
    if trainable_params:
        loss = (-surrogate_loss + baseline_loss)
        torch_backward(loss, retain_graph=True)
        # torch_backward(loss)
        # loss.backward()
    # import ipdb ; ipdb.set_trace()
    return loss

def PPO_update(optimizer, model, guide, policy, args):
    # weight = 1./num_particles
    # losses = []
    num_particles = args.nb_particles
    total_loss = 0.0
    old_policy = copy.deepcopy(policy)
    # model_trace, guide_trace = ELBO._get_traces(model, guide, policy,)
    guide_trace = pp.trace(guide).get_trace(policy, args)
    model_trace = pp.trace(pp.replay(model, trace=guide_trace)).get_trace(policy, args)
    guide_trace.compute_score_parts()
    model_trace.compute_log_prob()

    old_policy_log_prob = {name: site["score_parts"].score_function
                                for name, site in guide_trace.nodes.items() 
                                    if site["type"] == "sample"}
    # import ipdb; ipdb.set_trace()
    
    # with pp.trace(param_only=True) as param_capture:
    #     params = set(site["value"].unconstrained()
    #                     for site in param_capture.trace.nodes.values())
    params = set(site["value"].unconstrained() 
                    for name, site in guide_trace.nodes.items()
                        if site["type"] == "param")
    # import ipdb; ipdb.set_trace()
    
    for i in range(num_particles):
        # optimizer.zero_grad()
        zero_grads(params)
        loss = PPO_TraceGraph_loss(model_trace, guide_trace, old_policy_log_prob) # calls backward
        # losses.append(loss)
        total_loss += loss.item()
        # optimizer.step()
        optimizer(params)
        guide_trace = pp.trace(guide).get_trace(policy, args)
        guide_trace.compute_score_parts()
        model_trace = pp.trace(pp.replay(model, trace=guide_trace)).get_trace(policy, args)
        model_trace.compute_log_prob()
        # import ipdb; ipdb.set_trace()
    # the new policy should be in guide_trace 'W' 'b'
    #return new_policy
    return total_loss/num_particles