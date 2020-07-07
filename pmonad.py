""" Probability monad """
from __future__ import division
import random
from collections import Counter, namedtuple
from math import log, exp
import operator
import functools
import itertools
import sympy

import rfutils
from rfutils.compat import *

INF = float('inf')
_SENTINEL = object()

def keep_calling_forever(f):
    return iter(f, _SENTINEL)

# safelog : Float -> Float
def safelog(x):
    try:
        return log(x)
    except ValueError:
        return -INF

def identity(x):
    return x

# logaddexp : Float x Float -> Float
def logaddexp(one, two):
    return safelog(exp(one) + exp(two))

# logsumexp : [Float] -> Float
def logsumexp(xs):
    return safelog(sum(map(exp, xs)))

# reduce_by_key : (a x a -> a) x [(b, a)] -> {b -> a}
def reduce_by_key(f, keys_and_values):
    d = {}
    for k, v in keys_and_values:
        if k in d:
            d[k] = f(d[k], v)
        else:
            d[k] = v
    return d

def lazy_product_map(f, xs):
    """ equivalent to itertools.product(*map(f, xs)), but does not hold the values
    resulting from map(f, xs) in memory. xs must be a sequence. """
    if not xs:
        yield []
    else:
        x = xs[0]
        for result in f(x):
            for rest in lazy_product_map(f, xs[1:]):
                yield [result] + rest

class Monad(object):
    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.values) + ")"

    def __rshift__(self, f):
        return self.bind(f)

    def __add__(self, bindee_without_arg):
        return self.bind(lambda _: bindee_without_arg())

    # lift : (a -> b) -> (m a -> m b)
    @classmethod
    def lift(cls, f):
        @functools.wraps(f)
        def wrapper(a):
            return a.bind(cls.lift_ret(f))
        return wrapper

    # lift_ret : (a -> b) -> a -> m b
    @classmethod
    def lift_ret(cls, f):
        @functools.wraps(f)
        def wrapper(*a, **k):
            return cls.ret(f(*a, **k))
        return wrapper

    def bind_ret(self, f):
        return self.bind(self.lift_ret(f))

    @property
    def mzero(self):
        return type(self)(self.zero)

    @classmethod
    def guard(cls, truth):
        """ Usage:
        lambda x: Monad.guard(condition(x)) >> (lambda _: consequent(x))
        """
        if truth:
            return cls.ret(_SENTINEL) # irrelevant value
        else:
            return cls(cls.zero) # construct mzero

    @classmethod
    def mapM(cls, mf, xs):
        # Reference implementation to be overridden by something more efficient.
        def f(acc, x):
            # f acc x = do
            # r <- mf(x);
            # return acc + (r,)
            return mf(x).bind(lambda r: cls.ret(acc + (r,)))
        return cls.reduceM(f, xs, initial=())

class Amb(Monad):
    def __init__(self, values):
        self.values = values

    zero = []

    def sample(self):
        return next(iter(self))

    def bind(self, f):
        return Amb(rfutils.flatmap(f, self.values))

    @classmethod
    def ret(cls, x):
        return cls([x])

    def __iter__(self):
        return iter(self.values)

    # mapM : (a -> Amb b) x [a] -> Amb [b]
    @classmethod
    def mapM(cls, f, *xss):
        return Amb(itertools.product(*map(f, *xss)))

    # filterM : (a -> Amb Bool) x [a] -> Amb [a]
    @classmethod
    def filterM(cls, f, xs):
        return cls(itertools.compress(xs, mask) for mask in cls.mapM(f, xs))

    # reduceM : (a x a -> Amb a) x [a] -> Amb [a]
    @classmethod
    def reduceM(cls, f, xs, initial=None):
        def do_it(acc, xs):
            if not xs:
                yield acc
            else:
                x = xs[0]
                xs = xs[1:]
                for new_acc in nf(acc, x):
                    for res in do_it(new_acc, xs):
                        yield res
        xs = tuple(xs)
        if initial is None:
            return cls(do_it(xs[0], xs[1:]))
        else:
            return cls(do_it(initial, xs))

    def conditional(self, f=None, normalized=True):
        if f is None:
            f = lambda x: x

        class CDict(dict):
            def __missing__(d, key):
                samples = (y for x, y in map(f, self.values) if x == key)
                d[key] = Amb(samples)
                return d[key]

        return CDict()

def Samples(rf):
    return Amb(keep_calling_forever(rf))

Field = namedtuple('Field',
     ['add', 'sum', 'mul', 'div', 'zero', 'one', 'to_log', 'to_p', 'from_log', 'from_p', 'exp']
)
p_space = Field(
    operator.add, sum, operator.mul, operator.truediv, 0, 1, safelog, identity, exp, identity, operator.pow
)
log_space = Field(
    logaddexp, logsumexp, operator.add, operator.sub, -INF, 0, identity, exp, identity, safelog, operator.mul
)

class Enumeration(Monad):
    def __init__(self,
                 values,
                 marginalized=False,
                 normalized=False):
        self.marginalized = marginalized
        self.normalized = normalized
        self.values = values
        if isinstance(values, dict):
            self.marginalized = True
            self.values = values.items()
            self._dict = values
        else:
            self.values = values
            self._dict = None

    field = log_space
    zero = []

    def logp(self, x):
        return self.field.to_log(self.dict.get(x, self.field.zero))

    def p(self, x):
        return self.field.to_p(self.dict.get(x, self.field.zero))

    def bind(self, f):
        mul = self.field.mul
        def gen():
            for x, p_x in self.values:
                for y, p_y in f(x):
                    yield y, mul(p_y, p_x)
        return type(self)(gen()).marginalize().normalize()

    # return : a -> Enum a
    @classmethod
    def ret(cls, x):
        return cls(
            [(x, cls.field.one)],
            normalized=True,
            marginalized=True,
        )

    def marginalize(self):
        if self.marginalized:
            return self
        else:
            # add together probabilities of equal values
            result = reduce_by_key(self.field.add, self.values)
            # remove zero probability values
            zero = self.field.zero
            result = {k:v for k, v in result.items() if v != zero}
            return type(self)(
                result,
                marginalized=True,
                normalized=self.normalized,
            )

    def normalize(self):
        if self.normalized:
            return self
        else:
            enumeration = list(self.values)
            Z = self.field.sum(p for _, p in enumeration)
            div = self.field.div
            result = [(thing, div(p, Z)) for thing, p in enumeration]
            return type(self)(
                result,
                marginalized=self.marginalized,
                normalized=True,
            )

    def __iter__(self):
        return iter(self.values)

    @property
    def dict(self):
        if self._dict:
            return self._dict
        else:
            self._dict = dict(self.values)
            return self._dict

    def __getitem__(self, key):
        return self.dict[key]

    @classmethod
    def mapM(cls, ef, *xss):
        mul = cls.field.mul
        one = cls.field.one
        def gen():
            for sequence in itertools.product(*map(ef, *xss)):
                if sequence:
                    seq, ps = zip(*sequence)
                    yield tuple(seq), functools.reduce(mul, ps, one)
                else:
                    yield tuple(), one
        return cls(gen()).marginalize().normalize()

    @classmethod
    def reduceM(cls, ef, xs, initial=None):
        mul = cls.field.mul
        one = cls.field.one
        def do_it(acc, xs):
            if not xs:
                yield (acc, one)
            else:
                the_car = xs[0]
                the_cdr = xs[1:]
                new_acc_distro = ef(acc, the_car).marginalize().normalize()
                for new_acc, p in new_acc_distro:
                    for res, p_res in do_it(new_acc, the_cdr):
                        yield res, mul(p, p_res)
        xs = tuple(xs)
        if initial is None:
            result = do_it(xs[0], xs[1:])
        else:
            result = do_it(initial, xs)
        return cls(result).marginalize().normalize()

    def expectation(self, f):
        return sum(cls.field.to_p(lp)*f(v) for v, lp in self.values)

    def exponentiate(self, a):
        expo = self.field.exp
        return type(self)((value, expo(lp, a)) for value, lp in self.values).normalize()

    def entropy(self):
        return -sum(exp(logp)*logp for _, logp in self.normalize()) / log(2)

    def conditional(self, f=None, normalized=True):
        if f is None:
            f = lambda x: x

        add = self.field.add
        d = {}
        for value, p in self.values:
            condition, outcome = f(value)
            if condition in d:
                if outcome in d[condition]:
                    d[condition][outcome] = add(d[condition][outcome], p)
                else:
                    d[condition][outcome] = p
            else:
                d[condition] = {outcome: p}
        cls = type(self)
        if normalized:
            return {
                k : cls(v).normalize()
                for k, v in d.items()
            }
        else:
            return {k: cls(v) for k, v in d.items()}

    @classmethod
    def flip(cls, p=1/2):
        vals = [(True, safelog(p)), (False, safelog(1-p))]
        return cls(vals, normalized=True).marginalize()

    @classmethod
    def uniform(cls, xs):
        return cls((x, cls.field.one) for x in xs).marginalize().normalize()
        
class PSpaceEnumeration(Enumeration):
    field = p_space

    @classmethod
    def flip(cls, p=1/2):
        vals = [(True, p), (False, 1-p)]
        return cls(vals, normalized=True).marginalize()

    def entropy(self):
        return -sum(p*log(p) for _, p in self.normalize()) / log(2)

    def expectation(self, f):
        return sum(p*f(v) for v, p in self.values)


class SymbolicEnumeration(PSpaceEnumeration):

    def marginalize(self):
        result = super().marginalize()
        new_result = {k:sympy.simplify(v) for k, v in result.values}
        return type(result)(
            new_result,
            marginalized=True,
            normalized=result.normalized
        )

    def entropy(self):
        return -sum(p*sympy.log(p) for _, p in self.normalize())

def UniformEnumeration(xs):
    xs = list(xs)
    N = len(xs)
    return Enumeration([(x, -log(N)) for x in xs])

def UniformSamples(xs):
    return Samples(lambda: random.choice(xs))

def enumerator(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return Enumeration(f(*a, **k))
    return wrapper

def pspace_enumerator(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return PSpaceEnumeration(f(*a, **k))
    return wrapper

def uniform_enumerator(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return UniformEnumeration(f(*a, **k))
    return wrapper

deterministic = Enumeration.lift_ret
certainly = Enumeration.ret

def uniform(xs):
    xs = list(xs)
    n = len(xs)
    return Enumeration([(x, -log(n)) for x in xs])

def sampler(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return Samples(lambda: f(*a, **k))
    return wrapper

def enumeration_from_samples(samples, num_samples):
    counts = Counter(itertools.islice(samples, None, num_samples))
    return Enumeration((k, log(v)) for k, v in counts.items()).normalize()

def enumeration_from_sampling_function(f, num_samples):
    samples = iter(f, _SENTINEL)
    return enumeration_from_samples(samples, num_samples)

def approx_enumerator(num_samples):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*a, **k):
            sample_f = lambda: f(*a, **k)
            return enumeration_from_sampling_function(sample_f, num_samples)
        return wrapper
    return decorator

# enum_flip :: Float -> Enum Bool
@enumerator
def enum_flip(p):
    if p > 0:
        yield True, log(p)
    if p < 1:
        yield False, log(1-p)

@pspace_enumerator
def pspace_flip(p):
    if p > 0:
        yield True, p
    elif p < 1:
        yield False, 1 - p

def surprisal(distribution, value):
    return -distribution.field.to_log(distribution.dict[value])

def test_pythagorean_triples():
    n = 25
    result = uniform(range(1, n+1)) >> (lambda x: # x ~ uniform(1:n);
             uniform(range(x+1, n+1)) >> (lambda y: # y ~ uniform(x+1:n);
             uniform(range(y+1, n+1)) >> (lambda z: # z ~ uniform(y+1:n);
             Enumeration.guard(x**2 + y**2 == z**2) >> (lambda _: # constraint
             Enumeration.ret((x,y,z)))))) # return a triple deterministically
    assert set(result.dict.keys()) == {
        (3, 4, 5),
        (5, 12, 13),
        (6, 8, 10),
        (7, 24, 25),
        (8, 15, 17),
        (9, 12, 15),
        (12, 16, 20),
        (15, 20, 25)
    }
    assert all(logp == -2.079441541679836 for logp in result.dict.values())

def send_more_money():
    # Nice as an example but way too slow to be a test.
    
    def encode(*xs):
        return sum(10**i * x for i, x in enumerate(reversed(xs)))
    
    result = uniform(range(10)) >> (lambda s:
             uniform(range(10)) >> (lambda e:
             uniform(range(10)) >> (lambda n:
             uniform(range(10)) >> (lambda d:
             uniform(range(10)) >> (lambda m:
             uniform(range(10)) >> (lambda o:
             uniform(range(10)) >> (lambda r:
             uniform(range(10)) >> (lambda y:
             Enumeration.guard(
                 encode(s,e,n,d) + encode(m,o,r,e) == encode(m,o,n,e,y)
             ) >> (lambda _: Enumeration.ret((s,e,n,d,m,o,r,y)))))))))))

    assert result == (9, 5, 6, 7, 1, 0, 8, 2)
        
if __name__ == '__main__':
    import nose
    nose.runmodule()
