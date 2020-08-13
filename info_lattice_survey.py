import sys
import operator
import itertools

import numpy as np
import scipy.special
import rfutils
import dit

def enumerate_bitstrings(k):
    bools = itertools.product(*[range(2)]*k)
    for bool in bools:
        yield "".join(map(str, map(int, bool)))

def powerset(iterable, upto=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if upto is None:
        upto = len(s)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(upto+1)
    )        
def binary_source(K, sigma=0):
    probabilities = scipy.special.softmax(np.random.randn(2**K)*sigma)
    support = enumerate_bitstrings(K)
    d = dit.Distribution(dict(zip(support, probabilities)))
    names = ["x%d" % i for i in range(K)]
    d.set_rv_names(names)
    return d

flat = itertools.chain.from_iterable

def integer_partitions(n, k, l=1):
    '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
    # From https://stackoverflow.com/questions/18503096/python-integer-partitioning-with-given-k-partitions user Snakes and Coffee
    if k < 1:
        return
    elif k == 1:
        if n >= l:
            yield (n,)
        return
    else:
        for i in range(l, n+1):
            for result in integer_partitions(n-i, k-1, i):
                yield (i,) + result

def expansions(xs, k):
    for partition in integer_partitions(k, len(xs)):
        for partition_perm in set(itertools.permutations(partition)):
            expanded = [[x]*n for x, n in zip(xs, partition_perm)]
            yield tuple(flat(expanded))

def onto_mappings_to(codomain, K):
    return set(flat(map(itertools.permutations, expansions(codomain, K))))

def deterministic_binary_mappings(source, T=None):
    K = len(source.rvs)
    if T is None:
        T = K
    bitstrings = list(enumerate_bitstrings(T))
    source_names = sorted(source._rvs)
    signal_names = ["y%d" % i for i in range(T)]
    names = source_names + signal_names
    source_outcomes = source.outcomes
    source_pmf = source.pmf
    mappings = onto_mappings_to(bitstrings, 2**K)
    for mapping in mappings:
        pairs = list(zip(source_outcomes, mapping))
        d = dit.Distribution([outcome+bitstring for outcome, bitstring in pairs], source_pmf)
        d.set_rv_names(names)
        yield str(dict(pairs)), d

def is_weakly_systematic(mapping):
    # H[x_t | g_k] = 0 for all t for some permutation K of g.
    source_variables = [variable for variable in mapping._rvs if 'x' in variable]
    signal_variables = [variable for variable in mapping._rvs if 'y' in variable]
    def conditional_entropies():
        for perm in itertools.permutations(source_variables):
            reconstructions = [mapping.marginal([s, signal_variable]).condition_on([signal_variable])[-1] for s, signal_variable in zip(perm, signal_variables)]
            yield sum(sum(dit.shannon.entropy(c) for c in conds) for conds in reconstructions)
    return any(h==0 for h in conditional_entropies())

def measures(mapping):
    atoms = dict(information_lattice(mapping))
    result = {}
    for name, value in atoms.items():
        valence = (name.count('x'), name.count('y'))
        if valence[0] > 0 and valence[1] > 0:
            valence_label = "spectrum_%d_%d" % valence
            if valence_label in result:
                result[valence_label] += value
            else:
                result[valence_label] = value
    result.update(atoms)
    result['is_systematic'] = is_weakly_systematic(mapping)
    return result

def is_even(x):
    return x % 2 == 0

def information_lattice(d):
    for subset in powerset(d._rvs):
        if subset:
            name = "I_%s" % "".join(subset)
            vnames = [(v,) for v in subset]
            value = dit.multivariate.interaction_information(d, vnames)
            yield name, value

def survey(K, T=None, sigma=0):
    source = binary_source(K, sigma=sigma)
    for string, mapping in deterministic_binary_mappings(source, T=T):
        result = measures(mapping)
        result['mapping'] = string
        yield result

def main(K=3, T=3, sigma=1):
    if isinstance(K, str):
        K = int(K)
    if isinstance(T, str):
        T = int(T)
    if isinstance(sigma, str):
        sigma = float(sigma)
    assert K >= 0
    assert T >= 0
    assert sigma >= 0
    assert T <= K, "T must be less than or equal to K"
    rfutils.write_dicts(sys.stdout, survey(K, T=T, sigma=sigma))
    
if __name__ == '__main__':
    main(*sys.argv[1:])
    

    
    
    
        
