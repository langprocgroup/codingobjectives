""" Probabilistic context-free rewriting systems in the probability monad """
from __future__ import division
from collections import namedtuple, Counter
from math import log, exp
import functools
import operator

import rfutils
import pyrsistent as pyr

from pmonad import *

Rule = namedtuple('Rule', ['lhs', 'rhs'])
def concatenate(sequences):
    return sum(sequences, ())

def make_pcfg(monad, rules, start='S'):
    rewrites = process_pcfg_rules(monad, rules)
    return PCFG(monad, rewrites, start)

def make_bounded_pcfg(monad, rules, bound, start='S'):
    rewrites = process_pcfg_rules(monad, rules)
    return BoundedPCFG(monad, rewrites, start, bound)

def process_pcfg_rules(monad, rules):
    d = {}
    for rule, prob in rules:
        prob = monad.field.from_p(prob)
        if rule.lhs in d:
            d[rule.lhs].append((rule.rhs, prob))
        else:
            d[rule.lhs] = [(rule.rhs, prob)]
    for k, v in d.items():
        d[k] = monad(v).normalize()
    return d

def process_gpsg_rules(monad, rules):
    return process_pcfg_rules(monad, expand_gspg_rules(rules))


def dict_product(d):
    """ {k:[v]} -> [{k:v}] """
    canonical_order = sorted(d.keys())
    value_sets = [d[k] for k in canonical_order]
    assignments = itertools.product(*value_sets)
    for assignment in assignments:
        yield dict(zip(canonical_order, assignment))

def dict_subset(d, ks):
    return {k:d[k] for k in ks}

# [(a,b)] -> {a:{b}}
def dict_of_sets(pairs):
    return rfutils.mreduce_by_key(set.add, pairs, set)

def expand_gpsg_rules(rules):
    """ Compile GPSG-style rules into CFG rules.
    Syntax for a GPSG-style rule:
    A_{f} -> B_{f} C_{f}
    {f} is a copied feature. It is given a value by:
    A -> B_f:v. 
    Shared feature values must be introduced explicitly in multiple RHSs.
    I.e., A -> B_f:v1 C_f:v1
          A -> B_f:v2 C_f:v2. 
    """
    def free_variables_in(element):
        parts = element.split("_")
        for part in parts:
            if part.startswith("{") and part.endswith("}"):
                yield part.strip("{}")

    def possible_feature_values_in(element):
        parts = element.split("_")
        for part in parts:
            if ":" in part:
                k, v = part.split(":")
                yield k, part 
                
    def possible_feature_values(rules):
        elements = rfutils.flat((rule.lhs,) + rule.rhs for rule in rules)
        pairs = rfutils.flatmap(possible_feature_values_in, elements)
        return dict_of_sets(pairs)

    rules = list(rules) # we'll have to go through twice
    possibilities = possible_feature_values(rule for rule, _ in rules)
    for rule, prob in rules:
        free_variables = set(free_variables_in(rule.lhs))
        for element in rule.rhs:
            free_variables.update(free_variables_in(element))
        assignments = dict_product(
            dict_subset(possibilities, free_variables)
        )
        for assignment in assignments:
            new_lhs = rule.lhs.format_map(assignment)
            new_rhs = tuple(
                element.format_map(assignment) for element in rule.rhs
            )
            yield Rule(new_lhs, new_rhs), prob

def test_expand_gpsg_rules():
    stuff = expand_gpsg_rules([
        (Rule('S', ('NP_g:f', 'VP_g:f')), .5),
        (Rule('S', ('NP_g:m', 'VP_g:m')), .5),
        (Rule('NP_{g}', ('A_{g}', 'N_{g}')), 1)
    ])
    assert set(stuff) == {
        (Rule(lhs='S', rhs=('NP_g:f', 'VP_g:f')), 0.5),
        (Rule(lhs='S', rhs=('NP_g:m', 'VP_g:m')), 0.5),
        (Rule(lhs='NP_g:f', rhs=('A_g:f', 'N_g:f')), 1),
        (Rule(lhs='NP_g:m', rhs=('A_g:m', 'N_g:m')), 1),
    }

def process_pcfrs_rules(monad, rules):
    d = {}
    for rule, prob in rules:
        new_lhs = (rule.lhs, len(rule.rhs))
        prob = monad.field.from_p(prob)
        if rule.lhs in d:
            d[new_lhs].append((rule.rhs, prob))
        else:
            d[new_lhs] = [(rule.rhs, prob)]
    for k, v in d.items():
        d[k] = monad(v).normalize()
    return d

def make_pcfrs(monad, rules, start='S'):
    rewrites = process_pcfrs_rules(monad, rules)
    return PCFRS(monad, rewrites, start)

# PCFG : m x (a -> m [a]) x a
class PCFG(object):
    def __init__(self, monad, rewrites, start):
        self.monad = monad
        self.rewrites = rewrites
        self.start = start

    # rewrite_nonterminal : a -> m [a]
    def rewrite_nonterminal(self, symbol):
        return self.rewrites[symbol]

    # is_terminal : a -> Bool
    def is_terminal(self, symbol):
        return symbol not in self.rewrites

    # rewrite_symbol : a -> m [a]
    def rewrite_symbol(self, symbol):
        if self.is_terminal(symbol):
            return self.monad.ret((symbol,))
        else:
            return self.rewrite_nonterminal(symbol) >> (
                lambda string: self.monad.reduceM(
                    self.expand_and_combine,
                    string,
                    initial=()))

    # expand_and_combine : [a] x a -> m [a]
    def expand_and_combine(self, acc, symbol):
        return self.rewrite_symbol(symbol) >> (
            lambda part: self.monad.ret(acc + part))

    # distribution : m [a]
    def distribution(self):
        return self.rewrite_symbol(self.start)

# BoundedPCFG : m x (a -> m [a]) x a x Nat    
class BoundedPCFG(PCFG):
    """ PCFG where a symbol can only be rewritten n times recursively. 
    If symbols have indices (e.g., NP_i), the indices are ignored for the 
    purpose of counting symbols during derivations. """
    def __init__(self, monad, rewrites, start, bound):
        self.monad = monad
        self.rewrites = rewrites
        self.start = start
        self.bound = bound

    def rewrite_symbol(self, symbol, history):
        if self.is_terminal(symbol):
            return self.monad.ret((symbol,))
        else:
            symbol_bare = symbol.split("_")[0]
            condition = history.count(symbol_bare) <= self.bound
            new_history = history.add(symbol_bare)
            return self.monad.guard(condition) >> (
                lambda _: self.rewrite_nonterminal(symbol) >> (
                lambda string: self.monad.reduceM(
                    lambda a, s: self.expand_and_combine(a, s, new_history),
                    string,
                    initial=())))

    def expand_and_combine(self, acc, symbol, history):
        return self.rewrite_symbol(symbol, history) >> (
            lambda part: self.monad.ret(acc + part))

    def distribution(self):
        return self.rewrite_symbol(self.start, pyr.pbag([]))

def process_indexed_string(string):
    symbols = []
    part_of = []
    seen = {}
    for i, part in enumerate(string):
        if isinstance(part, str):
            symbols.append((part, 1))
            part_of.append(i)
        else:
            symbol, index, num_blocks = part
            symbols.append(symbol)
            if (symbol, index) in seen:
                part_of.append(seen[symbol, index])
            else:
                part_of.append(i)
                seen[symbol, index] = i
    return symbol, part_of
            
def put_into_indices(self, symbols, indices):
    seen = Counter()
    def gen():
        for index in indices:
            yield symbols[index][seen[index]]
            seen[index] += 1
    return tuple(gen())

class PCFRS(PCFG):
    # rules have format (symbol, num_blocks) -> (blocks)
    def __init__(self, monad, rewrites, start):
        self.monad = monad
        self.rewrites = rewrites
        self.start = start

    def expand_string(self, string):
        symbols, indices = process_indexed_string(string)
        return self.monad.mapM(self.rewrite_symbol, symbols) >> (
            lambda s: self.monad.ret(concatenate(put_into_indices(s, indices))))

    def distribution(self):
        return self.rewrite_nonterminal((self.start, 1))

def test_pcfg():
    from math import log, exp
    r1 = Rule('S', ('NP', 'VP'))
    r2 = Rule('NP', ('D', 'N'))
    r3 = Rule('VP', ('V', 'NP'))
    r4 = Rule('VP', ('V',))
    rules = [(r1, 1), (r2, 1), (r3, .25), (r4, .75)]
    pcfg = make_pcfg(Enumeration, rules)
    enum = pcfg.distribution()
    assert enum.dict[('D', 'N', 'V')] == log(.75), enum.dict
    assert enum.dict[('D', 'N', 'V', 'D', 'N')] == log(.25)
    assert sum(map(exp, enum.dict.values())) == 1

def test_bounded_pcfg():
    from math import log, exp
    r1 = Rule('S', ('a', 'S', 'b'))
    r2 = Rule('S', ())
    rules = [(r1, 1/2), (r2, 1/2)]
    
    pcfg = make_bounded_pcfg(Enumeration, rules, 1)
    enum = pcfg.distribution()
    assert enum.dict[('a', 'b')] == log(1/2)
    assert enum.dict[()] == log(1/2)

    pcfg = make_bounded_pcfg(Enumeration, rules, 2)
    enum = pcfg.distribution()
    assert enum.dict[()] == log(1/2)
    assert enum.dict[('a', 'b')] == log(1/4)
    assert enum.dict[('a', 'a', 'b', 'b')] == log(1/4)

    pcfg = make_bounded_pcfg(Enumeration, rules, 3)
    enum = pcfg.distribution()
    assert enum.dict[()] == log(1/2)
    assert enum.dict[('a', 'b')] == log(1/4)
    assert enum.dict[('a', 'a', 'b', 'b')] == log(1/8)
    assert enum.dict[('a', 'a', 'a', 'b', 'b', 'b')] == log(1/8)
    
def test_pcfrs():
    from math import log
    r1 = Rule('S', (('NP', 'VP'),))
    r2 = Rule('NP', (('D', 'N'),))
    r3 = Rule('NPR', (('D', 'N'), ('RP',)))
    r4 = Rule('VP', (('V',),))
    r5 = Rule('S', ((('NPR', 0, 0), 'VP', ('NPR', 0, 1)),))
    r6 = Rule('D', (('the',),))
    r7 = Rule('D', (('a',),))
    r8 = Rule('N', (('cat',),))
    r9 = Rule('N', (('dog',),))
    r10 = Rule('V', (('jumped',),))
    r11 = Rule('V', (('cried',),))
    r12 = Rule('RP', (('that I saw yesterday',),))
    r13 = Rule('RP', (('that belongs to Bob',),))

    rules = [
        (r1, log(3/4)),
        (r2, 0),
        (r3, 0),
        (r4, 0),
        (r5, log(1/4)),
        (r6, log(1/2)),
        (r7, log(1/2)),
        (r8, log(1/2)),
        (r9, log(1/2)),
        (r10, log(1/2)),
        (r11, log(1/2)),
        (r12, log(1/3)),
        (r13, log(2/3))
    ]

    pcfrs = make_pcfrs(Enumeration, rules)
    return pcfrs


if __name__ == '__main__':
    import nose
    nose.runmodule()
