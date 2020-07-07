The code in `codesynergy3.py` performs enumerations using the Probability monad defined in `pmonad.py`.

The module `pmonad.py` defines three probability monads: `Enumeration`, `PSpaceEnumeration`, and `SymbolicEnumeration`. `Enumeration` computes log probabilities, `PSpaceEnumeration` computes probabilities, and `SymbolicEnumeration` computes probabilities in terms of symbolic expressions using `sympy` symbols.



Here is an example of the use of an `Enumeration` object:

```{python}

# Define a probability distribution with support {ac, ad, bc, bd} and probabilities 1/2, 1/4, 1/8, 1/8:
X = PSpaceEnumeration({'aa': .5, 'ab': .25, 'ba': .125, 'bb': .125})

# Monad return defines a delta distribution. Define a deterministic distribution that puts all probability on 'a':
D = PSpaceEnumeration.ret('a')
# Result: D = PSpaceEnumeration([('a', 1)])

# Monad bind, represented as >>, samples from a distribution and applies a function to the sample, returning a new distribution.
Y = X >> (lambda x: PSpaceEnumeration.ret(x[0]))  # i.e., sample x ~ X, then apply the function that returns x[0] deterministically
# Result: Y = PSpaceEnumeration([('a', 0.625), ('b', 0.375)])

# Simple example of adding two distributions
 A = PSpaceEnumeration({0: 1/2, 1: 1/2})
 B = PSpaceEnumeration({1: 1/2, 2: 1/2})
 C = A >> (lambda a: B >> (lambda b: PSpaceEnumeration.ret(a+b)))
 # Result: C = PSpaceEnumeration([(1, 0.25), (2, 0.5), (3, 0.25)])

```