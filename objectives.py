import itertools

import rfutils
import torch

""" Optimzation approach to Huffman code """


# A code is a function G -> prob X
# So we can represent it as a stochastic G x X matrix L where L[g,x] = p(x|g).

# A code can also be represented incrementally, as p(x_t | g, x_{<t}) (the incremental policy representation).
# If T is the length of the longest code, then the incremental policy representation is a matrix
# G x X^T x (X+1)

DEFAULT_NUM_EPOCHS = 10000
EPSILON = 10 ** -12

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
    
    init = sigma * torch.randn(num_G, num_X)
    code = init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([code], **kwds)
    source = source[:, None]
    
    for i in range(num_epochs):
        # Loss is -I[X_t : G | X_{<t}] = < p(x_t, g | x_{<t}) / p(x_t | x_{<t}) p(g | x_{<t}) >
        opt.zero_grad()
        p_code = torch.softmax(code, dim=X_axis)
        joint = source * p_code
        increment = t.transform(joint)
        loss = J(joint, increment, i, i % print_every == 0)
        loss.backward()
        opt.step()
    return t, p_code


def J_huffman(incremental_weight=1, determinism_weight=0):
    def J(joint, G_C_Xt, i=None, verbose=False):
        marginal = joint.sum(axis=G_axis, keepdim=True)
        conditional = joint / marginal # p(g|x) = p(g, x) / p(x)
        H_G_given_X = -(joint * conditional.log()).sum()
        H_X = -(marginal * marginal.log()).sum()        

        C = G_C_Xt.sum((G_axis, Xt_axis), keepdim=True) + EPSILON
        G_C = G_C_Xt.sum(Xt_axis, keepdim=True) + EPSILON
        Xt_C = G_C_Xt.sum(G_axis, keepdim=True) + EPSILON
        I_Xt_G_given_C = (G_C_Xt * ((G_C_Xt + EPSILON).log() + C.log() - G_C.log() - Xt_C.log())).sum()
        
        loss = H_G_given_X - incremental_weight * I_Xt_G_given_C + determinism_weight * H_X
        if verbose:
            H_X_given_G = -(joint * (joint / joint.sum(X_axis, keepdim=True)).log()).sum()
            print("epoch =", i, "H[X|G] =", H_X_given_G.item(), "H[G|X] =", H_G_given_X.item(), "I[X_t : G | X_{<t}] = ", I_Xt_G_given_C.item(), "loss =", loss.item())
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
    def __init__(self, X):
        self.X = X # for example, [0, 1, 00, 01, 10, 11].
        contexts, V = zip(*flat(map(increments, self.X)))
        self.contexts = list(set(contexts))
        self.V = list(set(V))
        self.mask = torch.zeros(len(self.X), len(self.contexts), len(self.V))
        for i, x in enumerate(self.X):
            for context, v in increments(x):
                self.mask[i, self.contexts.index(context), self.V.index(v)] = increment_is_compatible(x, context, v)

    def transform(self, joint):
        # transform p(g, x) -> p(g, x_{<t}, x_t)
        prefix = (joint[:, :, None, None] * self.mask[None, :, :, :]).sum(axis=1) # G x V^(T-1) x V
        return prefix / prefix.sum() 

    
if __name__ == '__main__':
    source = torch.Tensor([1/2, 1/4, 1/8, 1/8])
    t, code = incremental_autoencoder(J_huffman(), source, 2, 3, sigma=.5)
    print(t.X)
    print(code)
    
