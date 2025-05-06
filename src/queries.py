import torch as T

from src.counterfactual import CTFTerm

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

def print_sample_probs(sample):
    for V in sample:
        values, probs = tensor_prob_dist(sample[V])
        for val, prob in zip(values, probs):
            print(f"{V}={val.item()}: Probability {prob}")

def get_u_n(ncm, u=None, n=10000):
    if u is None:
        U = ncm.pu.sample(n=n)
        return U, n
    else:
        n_new = len(u[next(iter(u))])
        return u, n_new

def get_conditioned_u(ncm, u=None, do={}, conditions=None, n=10000):
    u, n_new = get_u_n(ncm, u, n)
    if conditions is None: return u, n_new
    sample = ncm(u=u, evaluating=True)
    
    indices_to_keep = set(range(n_new))
    for c in conditions:
        itk = T.where(sample[c]==conditions[c])[0].tolist()
        indices_to_keep = indices_to_keep.intersection(set(itk))
    return {k:u[k][list(indices_to_keep)] for k in u}, len(indices_to_keep)

def sample_ctf(ncm, term: CTFTerm, conditions={}, n=10000, u=None, get_prob=True):
    U, n = get_conditioned_u(ncm, u, conditions=conditions, n=n)

    expanded_do_terms = dict()
    for k in term.do_vals:
        if k == "nested":
            nested_term = term.do_vals[k]
            ctf = sample_ctf(ncm=ncm, term=nested_term, n=n, u=U, get_prob=False)
            for var in nested_term.vars:
                expanded_do_terms.update({var: ctf[var]})
        else:
            expanded_do_terms[k] = expand_do(term.do_vals[k],n)

    return ncm(n=None, u=U, do=expanded_do_terms, select=term.vars, evaluating=True)

def prob_ctf(outcome, ncm, term: CTFTerm, conditions={}, n=10000, u=None, get_prob=True):
    sample = sample_ctf(ncm, term, conditions, n=n, u=u, get_prob=get_prob)
    return tensor_prob_dist(sample[outcome])

def cond_specific_queries(ncm, treatment, mediators, outcome, condition=None, u=None, n=10000):
    """
    type='x', 'z', or 'xz'. if type is z or xz must provide confounders
    If conditions, type not provided, will give x-specific for x0.

    Assumes treatment is binary.
    """
    U, _ = get_u_n(ncm, u, n)
    dox1 = {V: 1 for V in treatment}
    dox0 = {V: 0 for V in treatment}
    Y = {outcome}

    Wx0 = CTFTerm(mediators, dox0)
    Y_dox1wx0 = CTFTerm(Y, {**dox1, "nested": Wx0})
    values, PY_dox1wx0 = prob_ctf(outcome, ncm, Y_dox1wx0, condition, u=U)

    _, PY_dox0 = prob_ctf(outcome, ncm, CTFTerm(Y,dox0), condition, u=U)
    _, PY_dox1 = prob_ctf(outcome, ncm, CTFTerm(Y,dox1), condition, u=U)

    TE = PY_dox1 - PY_dox0
    DE = PY_dox1wx0 - PY_dox0
    IE = PY_dox1wx0 - PY_dox1
    return values, TE, DE, IE, U

def x_sym(ncm, treatment, mediators, outcome,  xvals=None, u=None, n=10000):
    U, _ = get_u_n(ncm, u, n)
    dox1 = {V: 1 for V in treatment}
    dox0 = {V: 0 for V in treatment}
    Y = {outcome}

    Wx0 = CTFTerm(mediators, dox0)
    Y_dox1wx0 = CTFTerm(Y, {**dox1, "nested": Wx0})
    values, PY_dox1wx0 = prob_ctf(outcome, ncm, Y_dox1wx0, xvals, u=U)

    Wx1 = CTFTerm(mediators, dox1)
    Y_dox0wx1 = CTFTerm(Y, {**dox0, "nested": Wx1})
    values, PY_dox0wx1 = prob_ctf(outcome, ncm, Y_dox0wx1, xvals, u=U)

    _, PY_dox0 = prob_ctf(outcome, ncm, CTFTerm(Y,dox0), xvals, u=U)
    _, PY_dox1 = prob_ctf(outcome, ncm, CTFTerm(Y,dox1), xvals, u=U)

    DEx0x1 = PY_dox1wx0 - PY_dox0
    DEx1x0 = PY_dox0wx1 - PY_dox1
    DEsym = 0.5 * (DEx0x1 - DEx1x0)

    IEx0x1 = PY_dox1wx0 - PY_dox1
    IEx1x0 = PY_dox0wx1 - PY_dox0
    IEsym = 0.5 * (IEx0x1 - IEx1x0)

    SE, _ = x_se(ncm, treatment, outcome, xvals=xvals, u=U)

    for i in range(values.size()[0]):
        y = round(values[i].item())
        print(f'x-DEsym({outcome}={y} | {xvals}) \t= {DEsym[i]:.4f}')
        print(f'x-IEsym({outcome}={y} | {xvals}) \t= {IEsym[i]:.4f}')
        print(f'x-SE({outcome}={y}) \t\t\t= {SE[i]:.4f}')
        print()
    return values, DEsym, IEsym, SE

def general_queries(ncm, treatment, mediators, outcome, u=None, n=10000):
    values, TE, DE, IE, U = cond_specific_queries(ncm, treatment, mediators, outcome, u=u, n=n)
    for i in range(values.size()[0]):
        y = round(values[i].item())
        print(f'TE ({outcome}={y}) \t= {TE[i]:.4f}')
        print(f'NDE ({outcome}={y}) \t= {DE[i]:.4f}')
        print(f'NIE ({outcome}={y}) \t= {IE[i]:.4f}')
        print()
    return U

def exp_se(ncm, treatment, outcome, xvals=None, u=None, n=10000):
    U, _ = get_u_n(ncm, u, n)
    if xvals is None: xvals = {V: 1 for V in treatment}

    values, pY_condx = prob_ctf(outcome, ncm, CTFTerm({outcome}), conditions=xvals, u=U)
    _, pY_dox = prob_ctf(outcome, ncm, CTFTerm({outcome}, do_vals=xvals), u=U)
    s1 = pY_condx.size()[0]
    s0 = pY_dox.size()[0]
    if s1 == 0: s1 = T.zeros(s0)
    if s0 == 0: s0 = T.zeros(s1)
    exp_se = pY_condx - pY_dox

    for i in range(values.size()[0]):
        y = round(values[i].item())
        print(f'Exp-SE_{xvals}({outcome}={y}) \t= {exp_se[i]:.4f}')
    return U

def x_se(ncm, treatment, outcome, xvals=None, u=None, n=10000):
    U, _ = get_u_n(ncm, u, n)
    if xvals is None: xvals = {V: 1 for V in treatment}
    
    x0 = {V: 0 for V in treatment}
    x1 = {V: 1 for V in treatment}
    _, PY_dox0_condx1 = prob_ctf(outcome, ncm, CTFTerm({outcome},x0), conditions=x1, u=U)
    _, PY_dox0_condx0 = prob_ctf(outcome, ncm, CTFTerm({outcome},x0), conditions=x0, u=U)
    s1 = PY_dox0_condx1.size()[0]
    s0 = PY_dox0_condx0.size()[0]
    if s1 == 0: s1 = T.zeros(s0)
    if s0 == 0: s0 = T.zeros(s1)
    SE = PY_dox0_condx1 - PY_dox0_condx0
    return SE, U

def x_specific_queries(ncm, treatment, mediators, outcome, xvals=None, u=None, n=10000):
    if xvals is None: xvals = {V: 1 for V in treatment}
    values, TE, DE, IE, U = cond_specific_queries(ncm, treatment, mediators, outcome, condition=xvals, u=u, n=n)

    # x0 = {V: 0 for V in treatment}
    # x1 = {V: 1 for V in treatment}
    # _, PY_dox0_condx1 = prob_ctf(outcome, ncm, CTFTerm({outcome},x0), conditions=x1, u=U)
    # _, PY_dox0_condx0 = prob_ctf(outcome, ncm, CTFTerm({outcome},x0), conditions=x0, u=U)
    # SE = PY_dox0_condx1 - PY_dox0_condx0
    SE, _ = x_se(ncm, treatment, outcome, xvals=xvals, u=U)

    for i in range(values.size()[0]):
        y = round(values[i].item())
        print(f'x-TE({outcome}={y} | {xvals}) \t= {TE[i]:.4f}')
        print(f'x-DE({outcome}={y} | {xvals}) \t= {DE[i]:.4f}')
        print(f'x-IE({outcome}={y} | {xvals}) \t= {IE[i]:.4f}')
        print(f'x-SE({outcome}={y}) \t\t\t= {SE[i]:.4f}')
        print()
    return U

def z_specific_queries(ncm, treatment, mediators, outcome, zvals, u=None, n=10000):
    assert zvals is not None
    values, TE, DE, IE, U = cond_specific_queries(ncm, treatment, mediators, outcome, condition=zvals, u=u, n=n)

    x1 = {V: 1 for V in treatment}
    _, PY_condx1z = prob_ctf(outcome, ncm, CTFTerm({outcome}), conditions={**x1, **zvals}, u=U)
    _, PY_dox1_condz = prob_ctf(outcome, ncm, CTFTerm({outcome},x1), conditions=zvals, u=U)
    SE = PY_condx1z - PY_dox1_condz

    for i in range(values.size()[0]):
        y = round(values[i].item())
        print(f'z-TE({outcome}={y} | {zvals}) \t= {TE[i]:.4f}')
        print(f'z-DE({outcome}={y} | {zvals}) \t= {DE[i]:.4f}')
        print(f'z-IE({outcome}={y} | {zvals}) \t= {IE[i]:.4f}')
        print(f'z-SE_{x1}({outcome}={y}) \t\t\t= {SE[i]:.4f}')
        print()
    return U

def xz_specific_queries(ncm, treatment, mediators, outcome, zvals, xvals=None, u=None, n=10000):
    assert zvals is not None
    if xvals is None: xvals = {V: 1 for V in treatment}
    values, TE, DE, IE, U = cond_specific_queries(ncm, treatment, mediators, outcome, condition={**xvals, **zvals}, u=u, n=n)

    x0 = {V: 0 for V in treatment}
    x1 = {V: 1 for V in treatment}
    _, PY_dox_condx1z = prob_ctf(outcome, ncm, CTFTerm({outcome},xvals), conditions={**x1, **zvals}, u=U)
    _, PY_dox_condx0z = prob_ctf(outcome, ncm, CTFTerm({outcome},xvals), conditions={**x0, **zvals}, u=U)
    SE = PY_dox_condx1z - PY_dox_condx0z

    for i in range(values.size()[0]):
        y = round(values[i].item())
        print(f'xz-TE({outcome}={y} | {xvals, zvals}) \t= {TE[i]:.4f}')
        print(f'xz-DE({outcome}={y} | {xvals, zvals}) \t= {DE[i]:.4f}')
        print(f'xz-IE({outcome}={y} | {xvals, zvals}) \t= {IE[i]:.4f}')
        print(f'xz-SE({outcome}={y}) \t\t\t= {SE[i]:.4f}')
        print()
    return U