import torch as T

def rounded(data, dec=None):
    """
    data: dictionary of the form tensor([[val1],[val2],[val3],...])
    vars: which data keys you want the rounded values for. If None, will do all of them.
    dec: number of decimal places. If None, will round to int.
    """
    return[round(i[0],dec) for i in data.tolist()]

def rounded_set(data, vars=None, dec=None):
    """
    Round the data values of the given variables to the specified number of decimal places.
    
    data: dictionary of the form {'key': tensor([[val1],[val2],[val3],...]),...}
    vars: which data keys you want the rounded values for. If None, will do all of them.
    dec: number of decimal places. If None, will round to int.

    returns a dictionary of the form: {'key': [rounded_val1, rounded_val2, rounded_val3,...],...}
    """
    if vars is None: vars = data.keys()
    return {V: rounded(data[V],dec) for V in vars}

def expand_do(val, n):
    """Kevin"""
    if T.is_tensor(val):
        return T.tile(val, (n, 1))
    else:
        return T.unsqueeze(T.ones(n) * val, 1)
    
def expanded_dos(vals,n):
    do_dict = {}
    for k in vals:
        do_dict[k] = expand_do(vals[k],n)
    return do_dict

def check_equal(input, val):
    """Kevin"""
    if T.is_tensor(val):
        return T.all(T.eq(input, T.tile(val, (input.shape[0], 1))), dim=1).bool()
    else:
        return T.squeeze(input == val)
    
def tensor_prob_dist(t):
    """Returns the probability distribution of a 1D tensor with discrete values."""
    unique_vals, counts = T.unique(t, return_counts=True)
    probs = counts.float() / counts.sum()
    return unique_vals, probs

def get_conditioned_u(ncm, u=None, do={}, conditions={}, n=10000):
    if u is None:
        u = ncm.pu.sample(n=n)
        n_new = n
    else:
        n_new = len(u[next(iter(u))])

    sample = ncm(u=u, evaluating=True)
    
    indices_to_keep = set(range(n_new))
    for c in conditions:
        itk = T.where(sample[c]==conditions[c])[0].tolist()
        indices_to_keep = indices_to_keep.intersection(set(itk))
    return {k:u[k][list(indices_to_keep)] for k in u}, len(indices_to_keep)

def compute_ctf(ncm, var, do={}, conditions={}, n=10000):
    U, n_new = get_conditioned_u(ncm, conditions=conditions, n=n)

    samples = dict()
    expanded_do_terms = dict()
    # print(do)
    for k in do:
        # print(k,do[k])
        if k == "nested":
            # TODO
            expanded_do_terms.update(compute_ctf("based on ctf term in v", get_prob=False))
        else:
            expanded_do_terms[k] = expand_do(do[k],n_new)

    samples = ncm(n=None, u=U, do=expanded_dos(do,n_new), evaluating=True)
    return tensor_prob_dist(samples[var])
