import picos
import numpy as np


def p_support_function(v, alpha, proxy_type=None):
    '''
    v must be numpy array of float64 datatype
    '''
    x_dim = len(v)

    P = picos.Problem()
    x = picos.RealVariable("x", x_dim)
    P.add_constraint(x.sum == 0)

    if proxy_type == 'inner_product':
        P.add_constraint(picos.Norm(x, p=2) <= alpha)
    elif proxy_type == 'l1_norm':
        P.add_constraint(picos.Norm(x, p=1) <= alpha)
    else:
        raise NotImplementedError

    P.set_objective("min", x | v)
    P.solve(verbosity=0, solver='cvxopt', primals=None)

    return P.value, np.squeeze(np.array(x.value))


def r_support_function_implicit(y, alpha):
    P = picos.Problem()
    x = picos.RealVariable("x")
    P.add_constraint(abs(x) <= alpha)

    P.set_objective("min", x*y)
    P.solve(verbosity=0)

    return P.value, x.value


def r_support_function(y, alpha):
    if y >= 0:
        return -alpha*y, -alpha
    else:
        return alpha*y, alpha


if __name__ == '__main__':
    p_support_function(v=np.array([5] * 100), alpha=0.5, proxy_type='inner_product')
    r_support_function_implicit(y=0.5, alpha=0.5)
    r_support_function(y=0.5, alpha=0.5)
