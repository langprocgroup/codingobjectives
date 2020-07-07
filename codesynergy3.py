import sys
import random
import functools
import itertools
import operator
from collections import Counter
from math import exp, log

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rfutils
import rfutils.ordering

from pmonad import *

HALT = "#"
EPSILON = 10 ** -10

def xorify(bit, M=PSpaceEnumeration):
    bit = int(bit)
    return M.flip(.5) >> (lambda b:
       M.ret((int(b), 1 - bit) if b else (int(b), bit)))

def xorify_sequence(x, M=PSpaceEnumeration):
    return M.mapM(lambda x: xorify(x, M), x) >> (lambda xs: M.ret(tuple(rfutils.flat(xs))))

def hadamard_code(x):
    n = len(x)
    masks = itertools.product((0,1), repeat=n)
    def gen():
        for mask in masks:
            bits = [x for x, m in zip(x, mask) if m]
            yield parity(bits)
    return tuple(gen())

def parity(bits):
    return sum(map(int, bits)) % 2

def shuffled(xs):
    ls = list(xs)
    random.shuffle(ls)
    return tuple(ls)

def huffman_example():
    G = {'a': .5, 'b': .25, 'c': .125, 'd': .125}
    C = {'a': "0", 'b': "10", 'c': "110", 'd': "111"}

    M = PSpaceEnumeration
    G_distro = M(G)
    C_distro = {g : M.ret(c) for g, c in C.items()}
    return G_distro, C_distro

def redundant_huffman_example1():
    G = {'a': .5, 'b': .25, 'c': .125, 'd': .125}
    C = {'a': "00", 'b': "1100", 'c': "111100", 'd': "111111"}

    M = PSpaceEnumeration
    G_distro = M(G)
    C_distro = {g : M.ret(c) for g, c in C.items()}
    return G_distro, C_distro

def redundant_huffman_example2():
    G = {'a': .5, 'b': .25, 'c': .125, 'd': .125}
    C = {'a': "00", 'b': "1010", 'c': "110110", 'd': "111111"}

    M = PSpaceEnumeration
    G_distro = M(G)
    C_distro = {g : M.ret(c) for g, c in C.items()}
    return G_distro, C_distro    

def simple_example():
    G = {'a': .25, 'b': .25, 'c': .25, 'd': .25}
    C = {'a': "00", 'b': "01", 'c': "10", 'd': "11"}

    M = PSpaceEnumeration
    G_distro = M(G)
    C_distro = {g : M.ret(c) for g, c in C.items()}

    return G_distro, C_distro

def non_prefix_free_example():
    G = {'a': .25, 'b': .25, 'c': .25, 'd': .25}
    C = {'a': "0", 'b': "1", 'c': "00", 'd': "01"}

    M = PSpaceEnumeration
    G_distro = M(G)
    C_distro = {g : M.ret(c) for g, c in C.items()}

    return G_distro, C_distro

def xor_source_code():
    M = PSpaceEnumeration
    G = M({'aa': .25, 'ab': .25, 'ba': .25, 'bb': .25})
    C = {
        'aa': M.ret('0'),
        'ab': M.ret('1'),
        'ba': M.ret('1'),
        'bb': M.ret('0'),
    }
    return G, C

def hadamard_example():
    G = {'a': .25, 'b': .25, 'c': .25, 'd': .25}
    C = {'a': "000", 'b': "011", 'c': "101", 'd': "110"}
    
    M = PSpaceEnumeration
    G_distro = M(G)
    C_distro = {g : M.ret(c) for g, c in C.items()}

    return G_distro, C_distro
    
def xor_example():
    G = {'a': .5, 'b': .5}
    M = PSpaceEnumeration
    G_distro = M(G)
    C_distro = {
        'a': M({'00': .5, '11': .5}),
        'b': M({'01': .5, '10': .5}),
    }
    return G_distro, C_distro

def xxor_example():
    # X-XOR. Hadamard of XOR.
    G = {'a': .5, 'b': .5}
    M = PSpaceEnumeration
    G_distro = M(G)
    C_distro = {
        'a': M({'000': .5, '110': .5}),
        'b': M({'011': .5, '101': .5}),
    }
    return G_distro, C_distro    

def geometric_example(t):
    # Source distribution is p(t) = 1/2^t.
    # Code is the Shannon code for this source.
    powers = [1]
    k = 2
    for _ in range(t):
        last = powers.pop()
        powers.extend([last*k]*k)
    G = {i : 1/p for i, p in enumerate(powers)}
    codes = ['1'*i + '0' for i in range(t+1)]
    codes[-1] = codes[-1][:-1] # pop the 0 off the end of the last one

    M = PSpaceEnumeration
    G_distro = M(G)
    C_distro = {i: M.ret(code) for i, code in enumerate(codes)}
    
    return G_distro, C_distro
            
def simple_compositional(num_factors, sigma=1):
    source = independent_binary_source(num_factors, sigma)
    M = type(source)
    source = source >> M.lift_ret(lambda x: "".join("ab"[c] for c in x))
    return source, {g : M.ret(tuple("ab".index(c) for c in g)) for g, _ in source}

def simple_noncompositional(num_factors, sigma=1):
    G, C = simple_compositional(num_factors, sigma=sigma)
    M = type(G)
    return G, dict(zip(C.keys(), shuffled(C.values())))

def double_independent_code(p1=.6, p2=.6):
    G = {
        'aa': p1*p2,
        'ab': p1*(1-p2),
        'ba': (1-p1)*p2,
        'bb': (1-p1)*(1-p2)
    }
    systematic_C = {
        'aa': "00",
        'ab': "01",
        'ba': "10",
        'bb': "11",
    }
    C2 = {
        'aa': "01",
        'ab': "00",
        'ba': "11",
        'bb': "10",        
    }
    C3 = {
        'aa': "00",
        'ab': "01",
        'ba': "11",
        'bb': "10",        
    }        
    
    M = PSpaceEnumeration
    return M(G), {g : M.ret(c) for g, c in systematic_C.items()}, {g : M.ret(c) for g, c in C2.items()}, {g : M.ret(c) for g, c in C3.items()}


def joint_incremental_transform(source, code, with_delimiter=True):
    # p(g)p(x|g) -> p(g, x<t)
    def gen():
        for g, p_g in source:
            x_dist = code[g]
            for sequence, p_x in x_dist:
                prefixes = rfutils.buildup(sequence)
                for prefix in prefixes:
                    yield (g, prefix), p_g * p_x
    M = type(source)                    
    return M(gen()).marginalize().normalize()

def test_joint_incremental_transform():
    source = PSpaceEnumeration({'a': .5, 'b': .5})
    code = {'a': PSpaceEnumeration.ret('0'),
            'b': PSpaceEnumeration.ret('11111')}
    ji = joint_incremental_transform(source, code)
    assert all(p == 1/6 for _, p in ji)

    source = PSpaceEnumeration({'a': 2/3, 'b': 1/3})
    ji = joint_incremental_transform(source, code)
    assert abs(ji['a', ('0',)] - 2/7) < EPSILON

def MI(joint):
    M = type(joint)
    one = joint >> M.lift_ret(lambda x: x[0])
    two = joint >> M.lift_ret(lambda x: x[1])
    return one.entropy() + two.entropy() - joint.entropy()

def conditional_entropy(joint):
    # H[Y|X] = H[X,Y] - H[X]
    M = type(joint)
    one = joint >> M.lift_ret(lambda x: x[0])
    return joint.entropy() - one.entropy()

def CMI(joint3):
    # Given joint distribution on (x,y,z), get I[X:Y|Z]
    M = type(joint3)
    three = joint3 >> M.lift_ret(lambda x: x[2])
    twothree = joint3 >> M.lift_ret(lambda x: (x[1], x[2]))
    onethree = joint3 >> M.lift_ret(lambda x: (x[0], x[2]))
    return onethree.entropy() + twothree.entropy() - joint3.entropy() - three.entropy()

def synergy(joint3):
    M = type(joint3)
    one = joint3 >> M.lift_ret(lambda x: x[0])
    two = joint3 >> M.lift_ret(lambda x: x[1])
    three = joint3 >> M.lift_ret(lambda x: x[2])
    onetwo = joint3 >> M.lift_ret(lambda x: (x[0], x[1]))
    twothree = joint3 >> M.lift_ret(lambda x: (x[1], x[2]))
    onethree = joint3 >> M.lift_ret(lambda x: (x[0], x[2]))
    return (
        onetwo.entropy() + twothree.entropy() + onethree.entropy()
        - one.entropy() - two.entropy() - three.entropy()
        - joint3.entropy()
    )

def powerset(iterable, upto=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if upto is None:
        upto = len(s)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(upto+1)
    )

def filter_indices(xs, indices):
    def gen():
        for i, x in enumerate(xs):
            if i in indices:
                yield x
    return tuple(gen())

def II_decomposition(triple, K):
    # decomposition of I[X:Y|Z] with respect to X.
    # I[X:Y|Z] = - \sum_k (-1)^k \sum_x_k I[*x_k:y|z]
    M = type(triple)
    
    def marginal(x, indices):
        g, x, c = x
        return filter_indices(g, indices), x, c
    
    def subset_distributions():
        for subset in list(powerset(range(K)))[1:]:
            yield subset, triple >> M.lift_ret(lambda x: marginal(x, subset))

    IIs = {}
    for subset, distro in subset_distributions():
        IIs[subset] = -(-1)**len(subset) * CMI(distro)
        for subsubset in powerset(subset, upto=len(subset)-1):
            if subsubset:
                IIs[subset] += (-1)**len(subset) * IIs[subsubset]

    return IIs

def information_spectrum(triple, K):
    d = II_decomposition(triple, K)
    result = [0]*K
    for subset, II in d.items():
        result[len(subset)-1] += II
    return result

def test_II_decomposition():
    G, C = xor_source_code()
    inc = joint_incremental_transform(G, C)
    triple = inc >> type(inc).lift_ret(lambda x: (x[0], x[1][-1], x[1][:-1]))
    decomp = II_decomposition(triple, 2)
    assert decomp[(0,)] == 0
    assert decomp[(1,)] == 0
    assert decomp[(0,1)] == 1

def incremental_components(source, code):
    def tripleify(x):
        g, x = x
        return g, x[-1], x[:-1]
    ji = joint_incremental_transform(source, code)
    M = type(ji)
    triple = ji >> M.lift_ret(tripleify)    
    syn = synergy(triple)
    m = CMI(triple) # I[X_t : G | X_{<t}] = I[X_t : G] + S[X_t : G : X_{<t}]
    H = (triple >> M.lift_ret(lambda x: x[-2])).entropy()
    I = MI(triple >> M.lift_ret(lambda x: (x[-2], x[-1]))) # not sure this is really entropy rate
    return m - syn, syn, H, I

def incremental_transform(code, with_delimiter=True):
    d = {}
    for g, x_dist in code.items():
        for sequence, p in x_dist:
            sequence = list(sequence)
            if with_delimiter:
                sequence = sequence + [HALT]
            prefixes = rfutils.buildup(sequence)
            for prefix in prefixes:
                *context, x = prefix
                context = tuple(context)
                if context in d:
                    if x in context:
                        d[g, context][x] = x_dist.field.add(d[context][x], p)
                    else:
                        d[g, context][x] = p
                else:
                    d[g, context] = {x: p}
        
    # Normalize the prefixes
    for prefix, prefix_distro in d.items():
        d[prefix] = type(x_dist)(prefix_distro).marginalize().normalize()

    return d

def marginalize_out(joint, which):
    def gen():
        for variables, p in joint:
            to_keep = tuple(x for i, x in enumerate(variables) if i not in which)
            yield to_keep, p
    M = type(joint)
    return M(gen()).marginalize().normalize()

def frozencounter(x):
    return tuple(Counter(x).items())

def iid(source, k):
    M = type(source)
    source = source.marginalize().normalize()
    return M.mapM(lambda _: source, range(k))

def unordered_iid(source, k):
    M = type(source)
    return iid(source, k) >> M.lift_ret(frozencounter)

def block_code(n):
    M = PSpaceEnumeration
    G_distro = iid(M.flip() >> M.lift_ret(int), n)
    C_distro = {c:M.ret(c) for c in G_distro.dict.keys()}
    return G_distro, C_distro

def repetition(xs, k):
    for x in xs:
        for _ in range(k):
            yield x

def repeated_block_code(n, k):
    M = PSpaceEnumeration
    G_distro = iid(M.flip() >> M.lift_ret(int), n)
    C_distro = {c: M.ret(tuple(repetition(c, k))) for c in G_distro.dict.keys()}
    return G_distro, C_distro

def bitstring(k):
    return random_string(k, [0,1])

def balanced_bitstring(k):
    assert k % 2 == 0, "k must be even"
    sequence = [0 for _ in range(k)] + [1 for _ in range(k)]
    random.shuffle(sequence)
    return "".join(map(str, sequence))

def random_string(k, sigma):
    bits = iter(lambda: random.choice(sigma), object())
    return tuple(map(str, rfutils.take(bits, k)))

def big_random_code(N_M=100, length=10, vocab=(0,1), seed=None):
    random.seed(seed)
    symbols = range(N_M)
    probs = np.ones(N_M)
    probs /= np.sum(probs)
    M = PSpaceEnumeration
    G = M(dict(zip(symbols, probs)))
    C = {symbol : M.ret(random_string(length, vocab)) for symbol in symbols}
    return G, C

def redundant_code(N_M=100, length=10, granularity=1):
    symbols = range(N_M)
    probs = np.ones(N_M)
    probs /= np.sum(probs)
    M = PSpaceEnumeration
    G = M(dict(zip(symbols, probs)))
    C = {symbol : M.ret((symbol,)*length) for symbol in symbols}
    return G, C

def logistic(x, k=1, x0=0):
    return 1 / (1 + np.exp(-k*(x - x0)))

def independent_binary_source(num_factors, sigma=1, M=PSpaceEnumeration):
    probabilities = logistic(np.random.randn(num_factors)*sigma)
    flips = M.mapM(PSpaceEnumeration.flip, probabilities)
    return flips

def enumerate_bitstrings(k):
    bools = itertools.product(*[range(2)]*k)
    for bool in bools:
        yield "".join(map(str, map(int, bool)))

def code_survey(num_factors, sigma=1, M=PSpaceEnumeration, full_context=False):
    # O(2^N!) -- very bad. and only 2 are systematic. 
    # For N=4, already 20,922,789,888,000
    source = independent_binary_source(num_factors, sigma=sigma, M=M)
    for strings in itertools.permutations(enumerate_bitstrings(num_factors)):
        code = {g : M.ret(strings[i]) for i, (g, p) in enumerate(source)}
        IG, Sf, H, I = incremental_components(source, code)
        ji = joint_incremental_transform(source, code)
        t_triple = ji >> M.lift_ret(lambda x: (x[0], x[1][-1], len(x[1][:-1])))
        full_triple = ji >> M.lift_ret(lambda x: (x[0], x[1][-1], x[1][:-1]))
        a1, a2, a3 = information_spectrum(full_triple, 3)
        a1t, a2t, a3t = information_spectrum(t_triple, 3)
        yield {
            'code': code,
            'IG': IG,
            'synergy': Sf,
            'H': H,
            'I': I,
            'a1_t': a1t,
            'a2_t': a2t,
            'a3_t': a3t,
            'a1_full': a1,
            'a2_full': a2,
            'a3_full': a3,            
            'strong_systematicity': is_strongly_systematic(source, code),
            'weak_systematicity': is_weakly_systematic(source, code),
        }

def all_same(xs):
    first = rfutils.first(xs)
    return all(x == first for x in xs)

def marginal_codes(source, code):
    M = type(source)
    T = len(rfutils.first(source.dict.keys()))    
    joint = source >> (lambda g: (code[g] >> (lambda x: M.ret((g,x)))))
    def marginal_code(t):
        # p(x_t | g_t)
        marginal = joint >> M.lift_ret(lambda x: (x[0][t], x[1][t]))
        return marginal
    return [marginal_code(t) for t in range(T)]

def is_weakly_systematic(source, code):
    # H[x_t | g_k] = 0 for all t for some permutation K of g.
    # implies I[x_t : x_{<t} | g_t] = 0
    M = type(source)
    T = len(rfutils.first(source.dict.keys()))
    code = {
        tuple(g): v
        for g, v in code.items()
    }
    def permutations(source):
        for perm in itertools.permutations(range(T)):
            yield source >> M.lift_ret(lambda g: tuple(rfutils.ordering.reorder(g, perm)))
    return any(
        all(conditional_entropy(m)==0 for m in marginal_codes(s, code))
        for s in permutations(source)
    )

def is_strongly_systematic(source, code):
    return all_same(tuple(dict(m).keys()) for m in marginal_codes(source, code))

def test_is_strongly_systematic():
    source = PSpaceEnumeration([
        ((True, True, True), 0.038065065132240845),
        ((True, True, False), 0.145274121524679),
        ((True, False, True), 0.15498811502842566),
        ((True, False, False), 0.591507256832453),
        ((False, True, True), 0.0028723949605722354),
        ((False, True, False), 0.010962404848628756),
        ((False, False, True), 0.011695424111573881),
        ((False, False, False), 0.044635217561426625)
    ])
    code = {
        (True, True, True): PSpaceEnumeration([('111', 1)]),
        (True, True, False): PSpaceEnumeration([('110', 1)]),
        (True, False, True): PSpaceEnumeration([('101', 1)]),
        (True, False, False): PSpaceEnumeration([('100', 1)]),
        (False, True, True): PSpaceEnumeration([('011', 1)]),
        (False, True, False): PSpaceEnumeration([('010', 1)]),
        (False, False, True): PSpaceEnumeration([('001', 1)]),
        (False, False, False): PSpaceEnumeration([('000', 1)])
    }
    assert is_strongly_systematic(source, code)
    code = {
        (True, True, True): PSpaceEnumeration([('111', 1)]),
        (True, True, False): PSpaceEnumeration([('110', 1)]),
        (True, False, True): PSpaceEnumeration([('101', 1)]),
        (True, False, False): PSpaceEnumeration([('100', 1)]),
        (False, True, True): PSpaceEnumeration([('011', 1)]),
        (False, True, False): PSpaceEnumeration([('010', 1)]),
        (False, False, True): PSpaceEnumeration([('000', 1)]),
        (False, False, False): PSpaceEnumeration([('001', 1)])
    }
    assert not is_strongly_systematic(source, code)
    

        
    
if __name__ == '__main__':
    pd.DataFrame(code_survey(3, sigma=0)).to_csv(sys.stdout)
