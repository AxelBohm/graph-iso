import numpy as np
from scipy.optimize import linprog, minimize
import itertools
import scipy.sparse as sp
from k_one_helpers import (construct_1_constraints, construct_rhs,
                           graph_to_array)

from weisfeiler_lehman import indexmap


def k_dim_linear_relax(G, H, k, solver="minimize"):
    """
    Solves the k-dimensional Surely-Adams relaxation for the graphs G and H.

    :solver: chooses the solver to solve linear system
    :returns: an isomorphism and the corresponding conclusion whether the two
            graphs are isomorphic or not.
    """

    n = G.number_of_nodes()
    m = H.number_of_nodes()
    if n != m:
        return (0, [])  # definitely not isomorphic

    (C, b) = construct_lp(G, H, k)

    (X, success) = solve_lin_prob(C, b, solver)

    (answer, isomorphism) = form_answer(X, success, n)

    return (answer, isomorphism)


def construct_lp(G, H, k):
    n = G.number_of_nodes()

    if k == 1:
        A = graph_to_array(G)
        B = graph_to_array(H)
        b = construct_rhs(n)
        C = construct_1_constraints(A, B)

    else:
        C = construct_k_constraints(G, H, k)
        numof_constraints = C.shape[0]
        b = np.concatenate([np.ones(2*n), np.zeros(numof_constraints-2*n)])

    return (C, b)


def form_answer(X, success, n):
    """ This function interprets the output of the solver.
    """

    # this is the candidate for an isomorphism
    isomorphism = X[:n**2].reshape(n, n)
    rounded_isomorphism = np.round(isomorphism, 2)

    if not success:
        # graphs are definitely not isomorphic
        answer = 0
    elif np.all(np.mod(X, 1) == 0):
        # The two graphs are isomorphic
        answer = 1
    elif np.all(np.mod(rounded_isomorphism, 1) == 0):
        # isomorphism up to numerical errors
        answer = 2
    else:
        # fractional isomorphism found
        answer = 3

    if answer == 2:
        return_iso = rounded_isomorphism
    elif answer == 0:
        return_iso = []
    else:
        return_iso = isomorphism

    return (answer, return_iso)


def solve_lin_prob(C, b, solver):

    dim = C.shape[1]

    if solver == "linear":
        # setting up LP
        obj = np.zeros(dim)
        res = linprog(obj, A_eq=C, b_eq=b, bounds=(0, 1),
                      method='interior-point',
                      # method='simplex',
                      options={'maxiter': 50, 'sparse': True})

        success = res.success
        X = res.x

    elif solver == "cvxopt":
        import cvxopt
        G = np.vstack([-np.eye(dim), np.eye(dim)])
        h = np.concatenate([np.zeros(dim), np.ones(dim)])
        sol = cvxopt.solvers.lp(cvxopt.matrix(obj), cvxopt.matrix(G),
                                cvxopt.matrix(h), cvxopt.matrix(C),
                                cvxopt.matrix(b))
        X = sol['x']

    elif solver == "qp":
        from quadprog import solve_qp
        X = solve_qp(C.T @ C, C.T @ b)

    elif solver == "minimize":

        # setting up general optimization problem
        def obj_fun(x):
            y = C @ x - b
            return 0.5 * np.dot(y, y)

        Q = C.T @ C
        CTb = C.T @ b
        bnds = ((0, 1),)*dim

        def gradient(x): return Q @ x - CTb

        opt = {'maxiter': 100}
        res = minimize(obj_fun, np.zeros(dim), method='L-BFGS-B',
                       bounds=bnds, jac=gradient, tol=10**(-14), options=opt)

        if np.round(res.fun, 2) > 0:
            success = False
        else:
            success = True

        X = res.x

    else:
        ValueError('solver not defined')

    return (X, success)


def construct_k_constraints(G, H, k):
    """combines all the different functions that contruct different constraints
    """
    n = G.number_of_nodes()
    C1 = construct_bijection_constraint(n, k)
    C2 = constraint_partial_isomorphism(n, k, G, H)

    C = sp.vstack([C1, C2])

    return C


def construct_bijection_constraint(n, k):
    overall_length = sum([n**(2*(j+1)) for j in range(k)])
    nrow = 2 * sum([n**(2*j + 1) for j in range(k)])
    C = sp.lil_matrix((nrow, overall_length))

    rowcount = 0
    for j in range(k):
        for v in range(n):
            tuples = itertools.product(*(range(n) for _ in range(2*j)))
            for pi in tuples:
                for w in range(n):
                    C[rowcount, flatten(pi + (v, w), n)] = 1
                    C[rowcount+1, flatten(pi + (w, v), n)] = 1

                if j > 0:
                    C[rowcount, flatten(pi, n)] = -1
                    C[rowcount+1, flatten(pi, n)] = -1
                rowcount += 2
    return C


def constraint_partial_isomorphism(n, k, G, H):
    overall_length = sum([n**(2*(j+1)) for j in range(k)])
    C = sp.lil_matrix((overall_length, overall_length))

    rowcount = 0
    for j in range(k):
        tuples = itertools.product(*(range(n) for _ in range(2*(j+1))))
        for tup in tuples:
            v = [tup[i] for i in range(0, 2*j + 2, 2)]
            w = [tup[i] for i in range(1, 2*j + 2, 2)]
            if not(is_partial_isomophism(v, w, G, H)):
                C[rowcount, flatten(tup, n)] = 1
                rowcount += 1

    return C[:rowcount]


def is_partial_isomophism(v, w, G, H):
    if is_partial_bijection(v, w) and preserves_edges(v, w, G, H):
        return True
    else:
        return False


def preserves_edges(v, w, G, H):
    j = len(v)
    for i in range(j):
        for l in range(i+1, j):
            if G.has_edge(v[i], v[l]) is not H.has_edge(w[i], w[l]):
                return False
    return True


def is_partial_bijection(v, w):
    j = len(v)
    for i in range(j):
        for l in range(i+1, j):
            if v[i] is v[l] and w[i] is not w[l]:
                return False
            elif v[i] is not v[l] and w[i] is w[l]:
                return False

    return True


def flatten(multiindex, n):
    j = int(len(multiindex)/2)
    dim_before = sum([n**(2*(l+1)) for l in range(j-1)])
    return indexmap(multiindex, n) + dim_before


