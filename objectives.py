import itertools
import operator
import functools
import random

import rfutils
import torch
import dit
import numpy as np
import matplotlib.pyplot as plt

import enumerate_lexicons as e

""" Optimization approach to systematic codes """

device = 'cpu'#torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Weirdly, the CPU is faster for these optimizations

# A code is a function G -> prob X
# So we can represent it as a stochastic G x X matrix L where L[g,x] = p(x|g).

# A code can also be represented incrementally, as p(x_t | g, x_{<t}) (the incremental policy representation).
# If T is the length of the longest code, then the incremental policy representation is a matrix
# G x X^T x (X+1)

DEFAULT_NUM_EPOCHS = 10000
EPSILON = 10 ** -12

colon = slice(None)

flat = itertools.chain.from_iterable

G_axis = 0
X_axis = 1
C_axis = 1
Xt_axis = 2

def incremental_autoencoder(J, source, V, K, sigma=1, num_epochs=DEFAULT_NUM_EPOCHS, print_every=1000, **kwds):
    """ J = H[G|X] + a I[X_t : G | X_{<t}] + b H[X_t | G, X_{<t}]. 
    Gives a Huffman code for a=1, b=0. 
    """
    num_G = source.shape[0]
    X = list(flat(itertools.product(*[range(V) for _ in range(k)]) for k in range(1, K+1)))
    num_X = len(X)
    t = IncrementalTransform(X)
    
    init = sigma * torch.randn(num_G, num_X).to(device)
    code = init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([code], **kwds)
    source = source[:, None]
    
    for i in range(num_epochs):
        # Loss is -I[X_t : G | X_{<t}] = < p(x_t, g | x_{<t}) / p(x_t | x_{<t}) p(g | x_{<t}) >
        opt.zero_grad()
        p_code = torch.softmax(code, dim=X_axis)
        joint = source * p_code
        loss = J(joint, t, i, i == num_epochs - 1)
        loss.backward()
        opt.step()
    return t, p_code

def zipf_plot(ps):
    ranked = ps.sort(descending=True).values.detach().numpy()
    ranks = np.arange(len(ranked))
    plt.plot(np.log(ranks), np.log(ranked))
    plt.xlabel("log rank")
    plt.ylabel("log probability")

def J_infomax(alpha=.5):
    """ J_infomax = H[G|X] + \alpha H[X] """
    def J(joint, _, i=None, verbose=False):
        marginal = joint.sum(axis=G_axis, keepdim=True)
        conditional = joint / marginal # p(g|x) = p(g, x) / p(x)
        H_G_given_X = -(joint * conditional.log()).sum()
        H_X = -(marginal * marginal.log()).sum()
        loss = H_G_given_X + alpha*H_X
        if verbose:
            H_X_given_G = -(joint * (joint / joint.sum(X_axis, keepdim=True)).log()).sum()
            print("epoch =", i,
                  "H[G|X] =", H_G_given_X.item(),
                  "H[X] =", H_X.item(),
                  "loss =", loss.item())
        return loss
    return J

def J_huffman2(incremental_weight=1, determinism_weight=.5):
    def J(joint, transform, i=None, verbose=False):
        marginal = joint.sum(axis=G_axis, keepdim=True)
        conditional = joint / marginal # p(g|x) = p(g, x) / p(x)
        H_G_given_X = -(joint * conditional.log()).sum()
        H_X = -(marginal * marginal.log()).sum()

        C_Xt = transform.transform(joint).sum(axis=G_axis)
        C = C_Xt.sum(axis=-1)
        H_Xt_given_C = -(C_Xt * (C_Xt + EPSILON).log()).sum() + (C * (C + EPSILON).log()).sum()

        loss = H_G_given_X + determinism_weight * H_X - incremental_weight * H_Xt_given_C
        if verbose:
            H_X_given_G = -(joint * (joint / joint.sum(X_axis, keepdim=True)).log()).sum()
            print("epoch =", i,
                  "H[G|X] =", H_G_given_X.item(),
                  "H[X_t | X_{<t}] = ", H_Xt_given_C.item(),
                  "H[X|G] =", H_X_given_G.item(),                  
                  "loss =", loss.item())
        return loss
    return J

def J_huffman(incremental_weight=3, determinism_weight=.5):
    """ J_huffman = H[G|X] - \alpha I[X_t : G | X_{<t}] + \beta H[X] """
   
    def J(joint, transform, i=None, verbose=False):
        marginal = joint.sum(axis=G_axis, keepdim=True)
        conditional = joint / marginal # p(g|x) = p(g, x) / p(x)
        H_G_given_X = -(joint * conditional.log()).sum()
        H_X = -(marginal * marginal.log()).sum()

        G_C_Xt = transform.transform(joint)
        C = G_C_Xt.sum((G_axis, Xt_axis), keepdim=True) + EPSILON
        G_C = G_C_Xt.sum(Xt_axis, keepdim=True) + EPSILON
        Xt_C = G_C_Xt.sum(G_axis, keepdim=True) + EPSILON
        I_Xt_G_given_C = (G_C_Xt * ((G_C_Xt + EPSILON).log() + C.log() - G_C.log() - Xt_C.log())).sum()
        
        loss = H_G_given_X - incremental_weight * I_Xt_G_given_C + determinism_weight * H_X
        if verbose:
            H_X_given_G = -(joint * (joint / joint.sum(X_axis, keepdim=True)).log()).sum()
            print("epoch =", i,
                  "H[G|X] =", H_G_given_X.item(),
                  "I[X_t : G | X_{<t}] = ", I_Xt_G_given_C.item(),
                  "H[X|G] =", H_X_given_G.item(),                  
                  "loss =", loss.item())
        return loss
    
    return J

# J = -I[G : X_{<t} : X_t] doesn't yield Huffman! Bad solution:
# p(g) = [1/2, 1/4, 1/4]
# code = [1, 0, 0]
# But it seems to work with H[G|X] - I[G : X_{<t} : X_]

# I[X:Y|Z] = < log p(x, y | z) / p(x|z) p(y|z) >
#          = < log p(x, y, z) p(z) / p(x, z) p(y, z) >
#          = < log p(x, y, z) + log p(z) - log p(x, z) - log p(y, z) >

def increment_is_compatible(string, context, x):
    context = tuple(context)
    string = tuple(string)
    return string[:len(context)] + (string[len(context)],) == context + (x,)

def test_increment_is_compatible():
    string = "abcdef"
    to_test = [
        ("ab", "c"),
        ("abcd", "e"),
    ]
    for c, x in to_test:
        assert increment_is_compatible(string, c, x)

huffman_joint = torch.Tensor([
    [1/2,0,0,0,0,0], # 0
    [0,0,0,0,0,0], # 1
    [0,0,0,0,0,0], # 00
    [0,0,0,0,0,0], # 01
    [0,0,0,0,1/4,0], # 10
    [0,0,0,0,0,1/4], # 11
]) # huffman code for [1/2, 1/4, 1/4] = [0, 10, 11]

def increments(xs):
    for *context, x in rfutils.buildup(xs):
        yield tuple(context), x

class IncrementalTransform:
    """ Keep track of the mapping from sequence indices to sequences """
    def __init__(self, X):
        self.X = X # for example, [0, 1, 00, 01, 10, 11].
        contexts, V = zip(*flat(map(increments, self.X)))
        self.contexts = list(set(contexts))
        self.V = list(set(V))
        self.mask = torch.zeros(len(self.X), len(self.contexts), len(self.V)).to(device)
        for i, x in enumerate(self.X):
            for context, v in increments(x):
                self.mask[i, self.contexts.index(context), self.V.index(v)] = increment_is_compatible(x, context, v)

    def transform(self, joint):
        """ transform p(..., x) -> p(..., x_{<t}, x_t) """
        num_dims = len(joint.shape) 
        joint_axes = (colon,) * num_dims + (None, None)
        mask_axes = (None,)*(num_dims-1) + (colon, colon, colon)
        prefix = (joint[joint_axes] * self.mask[mask_axes]).sum(axis=-3)
        return prefix / prefix.sum()

    def reconstruct(self, source, logq):
        # Requires that the X support is complete! Does not work for unbounded sequences!
        shape = tuple(source.shape) + (len(self.X),)
        logq_gsi = torch.empty(shape).to(device)
        for g, p_g in enumerate(source): # only works for 1D source...
            for i_s, s in enumerate(self.X):
                for i_inc, (c, x) in enumerate(increments(s)):
                    i_c = self.contexts.index(c)
                    i_x = self.V.index(x)
                    logq_gsi[g,i_s,i_inc] += logq[g, i_c, i_x]
        # q(g,s) = p(g) \prod q(x|g,c)
        logq_gs = logq_gsi.sum(-1)
        logq_gs += source.log()[:, None]
        return logq_gs    

def id_code_inc():
    return {
        'aa#0': 1/8,
        'aa#1': 0,
        'aa00': 1/8,
        'aa01': 0,
        'aa10': 0,
        'aa11': 0,

        'ab#0': 1/8,
        'ab#1': 0,
        'ab00': 0,
        'ab01': 1/8,
        'ab10': 0,
        'ab11': 0,

        'ba#0': 0,
        'ba#1': 1/8,
        'ba00': 0,
        'ba01': 0,
        'ba10': 1/8,
        'ba11': 0,

        'bb#0': 0,
        'bb#1': 1/8,
        'bb00': 0,
        'bb01': 0,
        'bb10': 0,
        'bb11': 1/8,                        
   }

def cnot_code_inc():
    return {
        'aa#0': 1/8,
        'aa#1': 0,
        'aa00': 1/8,
        'aa01': 0,
        'aa10': 0,
        'aa11': 0,

        'ab#0': 1/8,
        'ab#1': 0,
        'ab00': 0,
        'ab01': 1/8,
        'ab10': 0,
        'ab11': 0,

        'ba#0': 0,
        'ba#1': 1/8,
        'ba00': 0,
        'ba01': 0,
        'ba10': 0,
        'ba11': 1/8,

        'bb#0': 0,
        'bb#1': 1/8,
        'bb00': 0,
        'bb01': 0,
        'bb10': 1/8,
        'bb11': 0,                        

   }

def mi(p_xy, **kwds):
    p_x = p_xy.sum(axis=-2)
    p_y = p_xy.sum(axis=-1)
    return entropy(p_x) + entropy(p_y) - entropy(p_xy)

def prod(xs):
    return functools.reduce(operator.mul, xs, 1)

def predictive_ib(input_support, input_dist, bottleneck_dimension, tradeoff=1, sigma=1, num_epochs=DEFAULT_NUM_EPOCHS, print_every=1000, **kwds):
    """ J = -I[X_t : M_t] + tradeoff * I[M_t : X_{<t}] """
    C_dimension = 0 # context
    M_dimension = 1 # memory
    X_dimension = 2 # next symbol
    
    t = IncrementalTransform(input_support)
    p_CX = t.transform(input_dist)
    *_, C, X = p_CX.shape

    init = sigma * torch.randn(C, bottleneck_dimension)
    encoder = init.clone().detach().requires_grad_(True) # encoder is function C -> M
    opt = torch.optim.Adam([encoder], **kwds)
    
    for i in range(num_epochs):
        # Loss is -I[X_t : S_t] + tradeoff * I[S_t : X_{<t}]
        opt.zero_grad()
        p_encoder = torch.softmax(encoder, dim=M_dimension)
        joint = p_CX[:, None, :] * p_encoder[:, :, None] # shape C x M x X

        p_MX = joint.sum(dim=C_dimension)
        I_M_X = mi(p_MX)

        p_CM = joint.sum(dim=X_dimension)
        I_C_M = mi(p_CM)

        loss = -I_M_X + tradeoff * I_C_M
        
        loss.backward()
        opt.step()

        if i % print_every == 0:
            print("epoch = ", i, " I[M:X] = ", I_M_X.item(), " I[C:M] = ", I_C_M.item())
        
    return t, p_encoder

def star(sigma, k):
    return list(map("".join, itertools.product(*[sigma]*k)))

def cross_entropy(p, q, eps=EPSILON, *a, **k):
    return -(p * (q + eps).log()).sum(*a, **k)

def entropy(p, *a, **k):
    return cross_entropy(p, p, *a, **k)

def predictive_ib_code(which_code='opt', goal_k=2, signal_k=2, ragged=False, bottleneck_dimension=None, comm_weight=1, divergence_weight=1, complexity_weight=1, sigma=1, goal_V="ab", signal_V="01", num_epochs=DEFAULT_NUM_EPOCHS, print_every=1000, **kwds):
    # J = D_KL[p(x_t | g, s_t) || q(x_t | m_t)] + I_q[m_t : g, s_t]
    # The divergence term is equivalent to -I[x_t : m_t], which we will use
    # Source is uniform on aa, ab, ba, bb
    # It works! CNOT takes more memory than ID.
    source = torch.ones(*[len(goal_V)]*goal_k).to(device) / (len(goal_V)**goal_k)
    source_support = star(goal_V, goal_k)

    if ragged:
        signal_support = []
        for k in range(1, signal_k+1):
            signal_support.extend(star(signal_V, k))
        signal_support = [seq+"#" for seq in signal_support]
        signal_V += "#"
    else:
        signal_support = star(signal_V, signal_k)
        
    t = IncrementalTransform(signal_support)
    G = source.shape
    S = len(t.contexts)
    X = len(t.V)

    if bottleneck_dimension is None:
        Ms = S
    else:
        Ms = bottleneck_dimension
        
    erasure = "E"
    Mg_support = star(erasure + goal_V, goal_k)
    Mg = len(Mg_support)
    
    extractors = list(itertools.product(*[range(2)]*goal_k))    
    # p(m^g | g, e) : *V^k x 2^k x (V+1)^k    
    GEM = torch.empty(tuple(source.shape) + (len(extractors), len(Mg_support))).to(device)

    def is_compatible(g, extractor, Mg_element):
        return all(
            (e == 1 and m == goal_V[g]) or (e == 0 and m == erasure)
            for e, m, g in zip(extractor, Mg_element, g)
        )
        
    for indices, p_g in np.ndenumerate(source.cpu().numpy()):
        for i_e, extractor in enumerate(extractors):
            for i_m, Mg_element in enumerate(Mg_support):
                axes = tuple(indices) + (i_e, i_m)
                GEM[axes] = is_compatible(indices, extractor, Mg_element)


    code_shape = [len(goal_V)] * goal_k + [len(signal_support)]
    if which_code == 'opt':
        code_logit = torch.randn(*code_shape).to(device).clone().detach().requires_grad_(True)
    else:
        # depends on goal_k = 2
        code = torch.zeros(code_shape).to(device)
        if which_code == "cnot":
            which_code = {
                'aa': '00',
                'ab': '01',
                'ba': '11',
                'bb': '10',
            }
        elif which_code == "id":
            which_code = {
                'aa': '00',
                'ab': '01',
                'ba': '10',
                'bb': '11',
            }
        elif which_code == 'id3':
            which_code = {
                'aaa': '000',
                'aab': '001',
                'aba': '010',
                'abb': '011',
                'baa': '100',
                'bab': '101',
                'bba': '110',
                'bbb': '111',
            }
        elif which_code == 'cnot3':
            which_code = {
                'aaa': '000',
                'aab': '001',
                'aba': '011',
                'abb': '010',
                'baa': '100',
                'bab': '101',
                'bba': '111',
                'bbb': '110',
            }            
        elif which_code == 'random3':
            strings = star("01", 3)
            random.shuffle(strings)
            which_code = {
                g : s for g, s in zip(source_support, strings)
            }
        elif which_code == 'words2':
            assert ragged
            which_code = {
                'aa': '000#',
                'ab': '011#',
                'ba': '100#',
                'bb': '111#',
            }
        elif which_code == 'words2-efficient':
            assert ragged
            which_code = {
                'aa': '0#',
                'ab': '01#',
                'ba': '1#',
                'bb': '11#',
            }                
        elif which_code == 'words2-tangled':
            assert ragged
            which_code = {
                'aa': '000#',
                'ab': '011#',
                'ba': '100#',
                'bb': '101#',
            }
        elif which_code == 'words2-interleaved':
            assert ragged
            which_code = {
                'aa': '000#',
                'ab': '101#',
                'ba': '010#',
                'bb': '111#',
            }
        elif which_code == 'paradigm_systematic':
            assert not ragged
            assert goal_V == "abc"
            which_code = {
                'aa': '00',
                'ab': '01',
                'ac': '01',
                'ba': '10',
                'bb': '11',
                'bc': '11',
                'ca': '20',
                'cb': '21',
                'cc': '21',
            }
        elif which_code == 'paradigm_nonsystematic':
            assert not ragged
            assert goal_V == "abc"
            which_code = {
                'aa': '00',
                'ab': '01',
                'ac': '01',
                'ba': '10',
                'bb': '11',
                'bc': '11',
                'ca': '20',
                'cb': '20',
                'cc': '21',
            }

        assert goal_k == len(next(iter(which_code.keys())))
        
        for g, x in which_code.items():
            index = tuple(goal_V.index(gi) for gi in g) + (signal_support.index(x),)
            code[index] = 1

        p_gs = source.unsqueeze(-1) * code
        p = t.transform(p_gs) # p(...g..., s_t, x_t)
        p_context = p.sum(axis=-1, keepdim=True)
        p_expanded = p.unsqueeze(-2).unsqueeze(-2)
        p_conditional = p / (p_context+EPSILON)

    # m_t = <f(s_t), h(g, s_t)>
    q_state_mem_logit = (sigma*torch.randn(S, Ms)).to(device).clone().detach().requires_grad_(True)
    q_goal_extractor_logit = (sigma*torch.randn(S, 2**goal_k)).to(device).clone().detach().requires_grad_(True)

    if which_code == 'opt':
        opt = torch.optim.Adam([q_state_mem_logit, q_goal_extractor_logit, code_logit], **kwds)
    else:
        opt = torch.optim.Adam([q_state_mem_logit, q_goal_extractor_logit], **kwds)        
        

    *G_axes, S_axis, Mg_axis, Ms_axis, X_axis = range(4 + goal_k)
    G_axes = tuple(G_axes)

    for i in range(num_epochs):
        opt.zero_grad()

        if which_code == 'opt':
            code = torch.softmax(code_logit, -1)
            p_gs = source.unsqueeze(-1) * code
            p = t.transform(p_gs) # p(...g..., s_t, x_t)
            p_context = p.sum(axis=-1, keepdim=True)
            p_expanded = p.unsqueeze(-2).unsqueeze(-2)
            p_conditional = p / p_context

        code_mi = mi(p_gs.reshape(prod(G), len(signal_support)))
        I_gsx = mi(p.reshape(prod(G)*S, X))

        # q(m^s_t|s_t) is a simple discrete bottleneck
        q_state_mem = torch.softmax(q_state_mem_logit, -1)[(None,)*goal_k + (colon, None, colon, None)] # shape 1 x 1 x S x 1 x Ms x 1

        # q(m^g|g, s_t) is a bit more involved
        q_goal_extractor = torch.softmax(q_goal_extractor_logit, -1) # shape S x E
        # p(m_g | g, s) = \sum_e p(m_g | g, e) q(e|s_t)
        q_goal_mem = (q_goal_extractor[(None,)*goal_k + (colon, colon, None)] * GEM.unsqueeze(-3)).sum(axis=-2).unsqueeze(-1).unsqueeze(-1) # sum out E

        # q(g, s_t, m^g, m^s_t, x_t) = p(g, s_t, x_t) q(m^g | g, s_t) q(m^s_t | s_t)
        q_full = p_expanded * q_state_mem * q_goal_mem
        # G1 x G2 x S x Mg x Ms x X

        # I[m_t : x_t]
        q_mx = q_full.sum(dim=G_axes + (S_axis,)).reshape(Mg*Ms, X) # shape M x X
        I_mx = mi(q_mx) 

        # I[m_t : s_t, g] -- maybe split this up in the future.
        q_sm = q_full.sum(dim=X_axis).reshape(prod(G)*S, Mg*Ms)
        I_sm = mi(q_sm)

        # I[m^s_t : s_t]
        q_sms = q_full.sum(dim=G_axes + (X_axis,)).reshape(S, Mg*Ms)
        I_sms = mi(q_sms)

        # I[m^g : g]
        q_gmg = q_full.sum(dim=(S_axis, X_axis)).reshape(prod(G), Mg*Ms)
        I_gmg = mi(q_gmg)

        S_mgs = I_sm - I_sms - I_gmg 

        q_m = q_full.sum(axis=G_axes + (S_axis, X_axis)).reshape(Mg*Ms)
        H_m = entropy(q_m)

        if not hasattr(complexity_weight, '__iter__'):
            complexity_weight = [complexity_weight]*3
        elif hasattr(complexity_weight, 'shape') and len(complexity_weight.shape) == 0:
            complexity_weight = [complexity_weight]*3
                
        complexity = complexity_weight[0] * I_sms + complexity_weight[1] * I_gmg + complexity_weight[2] * S_mgs

        # directly calculate divergence
        # \sum_{g,s,x} p(g,s,x) \log p(x|g,s)/p(x|m)
        q_conditional = (q_mx / q_m[:, None])[None, :, :] # 1 x M x X
        q_joint = q_full.reshape(prod(G)*S, Mg*Ms, X) # GS x M x X
        divergence = cross_entropy(q_joint, q_conditional) - cross_entropy(p, p_conditional)
        
        loss = divergence_weight * divergence + complexity - comm_weight * code_mi

        loss.backward()
        opt.step()

        if print_every is not None and i % print_every == 0:
            print("epoch = ", i, " code MI = ", code_mi.item(), " divergence = ", divergence.item(), " I_gmg = ", I_gmg.item(), " I_sms = ", I_sms.item(), " S_mgs = ", S_mgs.item(), " I_gsm = ", I_sm.item())

    return source, code, q_state_mem, q_goal_extractor, code_mi.item(), I_sm.item(), divergence.item(), t

def run_codes(codes, num_restarts=1, max_attempts=5, tol=0.005, complexity_weight=1, comm_weight=.2, **kwds):
    def run_with_restarts(c):
        for r in range(num_restarts):
            for a in range(max_attempts):
                code, _, _, mi, complexity, divergence, t = predictive_ib_code(which_code=c, **kwds)
                if divergence < tol:
                    break
                else:
                    print("Rejected with divergence %s" % divergence)
            loss = divergence + complexity_weight*complexity - comm_weight*mi
            yield loss, code, mi, complexity, divergence, t
    for c in codes:
        _, code, mi, complexity, _, t = min(run_with_restarts(c))
        hcode = hard_code(code, t)
        yield hcode, mi, complexity

def hard_code(code, t):
    indices = code.argmax(-1)
    return {g:t.X[i] for g, i in np.ndenumerate(indices.cpu().numpy())}

def is_systematic(code):
    # H[x_t | g_k] = 0 for all t for some permutation K of g. (what if T>K?)
    K = len(next(iter(code.keys())))
    T = len(next(iter(code.values())))
    Z = len(code)
    
    mapping = dit.Distribution({tuple(key)+tuple(val):1/Z for key, val in code.items()})
    
    def conditional_entropies():
        for perm in e.onto_mappings(T, K):
            reconstructions = [
                mapping.marginal([s, signal_variable]).condition_on([0])[-1]
                for s, signal_variable in zip(perm, range(K, K+T))
            ]
            yield sum(sum(dit.shannon.entropy(c) for c in conds) for conds in reconstructions)
            
    return any(h==0 for h in conditional_entropies())


def greedy_example(tradeoff=1, sigma=1, timepref=1, num_epochs=10000, print_every=1000, **kwds):
    # we get the easy-first behavior when using the old wrong way of computing q(x)
    # not when using the new way with whole-kl or incremental-kl
    # can we introduce future discounting?
    p_a = .5
    K = 2
    source = torch.Tensor([p_a, 1 - p_a])
    support = star("abcd", K)    
    code = torch.zeros(len(source), len(support))
    code[0, support.index("ab")] = 1/2
    code[0, support.index("ba")] = 1/2
    code[1, support.index("cd")] = 1/2
    code[1, support.index("dc")] = 1/2
    
    costs = torch.Tensor([8,4,2,1])
    t = IncrementalTransform(support)
    p_gs = source[:, None] * code
    H_gs = entropy(p_gs)
    p = t.transform(p_gs) # p(g, x_{<t}, x_t)
    
    conditional_p = p / p.sum(axis=-1, keepdim=True)
    conditional_p[torch.isnan(conditional_p)] = 0
    H_p = cross_entropy(p, conditional_p).item()

    logp_gc = p.sum(-1, keepdim=True).log()
    
    init = sigma * torch.randn(len(source), len(t.contexts), len(t.V))
    q_logit = init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([q_logit], **kwds)

    def reconstruct(logq):
        logq_gsi = torch.zeros(len(source), code.shape[-1], code.shape[-2])
        for g, p_g in enumerate(source):
            for i_s, s in enumerate(t.X):
                for i_inc, (c, x) in enumerate(increments(s)):
                    i_c = t.contexts.index(c)
                    i_x = t.V.index(x)
                    logq_gsi[g,i_s,i_inc] += logq[g, i_c, i_x]
                    # q_gs[g,s] = p(g) \prod q(x|g,c)
        logq_gs = logq_gsi.sum(-1)
        logq_gs += source.log()[:, None]
        return logq_gs

    for i in range(num_epochs):
        opt.zero_grad()
        logq = torch.log_softmax(q_logit, dim=-1) # q(x_t | g, x_{<t})
        incremental_cross_ent = -(p * logq).sum()
        incremental_m_proj = cross_entropy(logq.exp(), p)
        # calculating expected cost is a little tricky because it requires q(x_t) = \sum_{x_{<t}} q(x_{<t}, x_t)
        # this requires undoing the incremental transform.
        # for now, let's approximate it as q(x_t) = \sum_{g, x_{<t}} p(g, x_{<t}) q(x_t|x_{<t})
        logq_x_old = (logp_gc + logq).logsumexp((0,1))

        # Or, let's do it right. q(x_{<t} = #) = 1/2
        # Need q(g, x_{<t}) = p(g)q(x_{<t}|g)
        logq_gs = reconstruct(logq) # q(g, s)
        whole_cross_ent = -(p_gs * logq_gs).sum()
        whole_m_proj = cross_entropy(logq_gs.exp(), p_gs)
        logq_x = t.transform(logq_gs.exp()).sum(axis=(0,1)).log()
        expected_cost = (logq_x.exp() * costs).sum() 
        
        J = incremental_cross_ent + tradeoff * expected_cost
        J.backward()
        opt.step()
        if i % print_every == 0:
            inc_kl = incremental_cross_ent - H_p
            whole_kl = whole_cross_ent - H_gs
            print("epoch = ", i, " incKL = ", inc_kl.item(), " wholeKL = ", whole_kl.item(), " <C> = ", expected_cost.item())

    return t, logq.exp(), conditional_p, logq_gs.exp(), p_gs

def predictive_ib_example():
    support = ['aa', 'ba', 'cc', 'dc'] # so we should map {a,b} and {c,d} to two memory symbols
    distro = torch.ones(4)/4
    t, q = predictive_ib(support, distro, 3, tradeoff=.1) # seems to work

# The Recursive Information Bottleneck is basically handled by pfa.py...
    
def huffman_example():
    source = torch.Tensor([1/2, 1/4, 1/8, 1/8]).to(device)
    t, code = incremental_autoencoder(J_huffman(), source, 2, 4, sigma=.5)
    return t, code

def pib_code_parameter_sweep(granularity=50, num_runs=10, **kwds):
    # Each optimization takes 30s
    alphas = torch.linspace(0,1,granularity)
    betas = torch.linspace(0,1,granularity)
    for alpha in alphas:
        for beta in betas:
            for r in range(num_runs):
                source, code, _, _, mi, complexity, divergence, t = predictive_ib_code(complexity_weight=alpha, comm_weight=beta, print_every=None, **kwds)
                hcode = hard_code(code, t)
                nondeterminism = cross_entropy(source.unsqueeze(-1)*code, code).item()
                yield {
                    'alpha': alpha.item(),
                    'beta': beta.item(),
                    'run': r,
                    'mi': mi,
                    'complexity': complexity,
                    'divergence': divergence,
                    'systematic': is_systematic(hcode),
                    'hard_code': hcode,
                    'nondeterminism': nondeterminism,
                }
    
    
