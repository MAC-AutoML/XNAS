import torch
import torch.nn.functional as F


def optimizer_transfer(optimizer_old, optimizer_new):
    for i, p in enumerate(optimizer_new.param_groups[0]['params']):
        if not hasattr(p, 'raw_id'):
            optimizer_new.state[p] = optimizer_old.state[p]
            continue
        state_old = optimizer_old.state_dict()['state'][p.raw_id]
        state_new = optimizer_new.state[p]

        state_new['momentum_buffer'] = state_old['momentum_buffer']
        if p.t == 'bn':
            # BN layer
            state_new['momentum_buffer'] = torch.cat(
                [state_new['momentum_buffer'], state_new['momentum_buffer'][p.out_index].clone()], dim=0)
            # clean to enable multiple call
            del p.t, p.raw_id, p.out_index

        elif p.t == 'conv':
            # conv layer
            if hasattr(p, 'in_index'):
                state_new['momentum_buffer'] = torch.cat(
                    [state_new['momentum_buffer'], state_new['momentum_buffer'][:, p.in_index, :, :].clone()], dim=1)
            if hasattr(p, 'out_index'):
                state_new['momentum_buffer'] = torch.cat(
                    [state_new['momentum_buffer'], state_new['momentum_buffer'][p.out_index, :, :, :].clone()], dim=0)
            # clean to enable multiple call
            del p.t, p.raw_id
            if hasattr(p, 'in_index'):
                del p.in_index
            if hasattr(p, 'out_index'):
                del p.out_index
    print('%d momemtum buffers loaded' % (i+1))
    return optimizer_new

def scheduler_transfer(scheduler_old, scheduler_new): 
    scheduler_new.load_state_dict(scheduler_old.state_dict())
    print('scheduler loaded')
    return scheduler_new


def process_step_vector(x, method, mask, tau=None):
    if method == "softmax":
        output = F.softmax(x, dim=-1)
    elif method == "dirichlet":
        output = torch.distributions.dirichlet.Dirichlet(F.elu(x) + 1).rsample()
    elif method == "gumbel":
        output = F.gumbel_softmax(x, tau=tau, hard=False, dim=-1)

    if mask is None:
        return output
    else:
        output_pruned = torch.zeros_like(output)
        output_pruned[mask] = output[mask]
        output_pruned /= output_pruned.sum()
        assert (output_pruned[~mask] == 0.0).all()
        return output_pruned


def process_step_matrix(x, method, mask, tau=None):
    weights = []
    if mask is None:
        for line in x:
            weights.append(process_step_vector(line, method, None, tau))
    else:
        for i, line in enumerate(x):
            weights.append(process_step_vector(line, method, mask[i], tau))
    return torch.stack(weights)


def prune(x, num_keep, mask, reset=False):
    if not mask is None:
        x.data[~mask] -= 1000000
    src, index = x.topk(k=num_keep, dim=-1)
    if not reset:
        x.data.copy_(torch.zeros_like(x).scatter(dim=1, index=index, src=src))
    else:
        x.data.copy_(
            torch.zeros_like(x).scatter(
                dim=1, index=index, src=1e-3 * torch.randn_like(src)
            )
        )
    mask = torch.zeros_like(x, dtype=torch.bool).scatter(
        dim=1, index=index, src=torch.ones_like(src, dtype=torch.bool)
    )
    return mask
