import networkx as nx
import numpy as np


def construct_rhs(n):
    ones = [1] * 2*n
    zeros = [0] * n**2
    return ones+zeros


def graph_to_array(G):
    A = nx.adjacency_matrix(G)
    return A.toarray()


def construct_edge_preserving_constraint(A, B):
    """ for X to be a graph isomorphism it must hold that: if v1 is mapped to
    w1 and v2 is mapped to w2, and if there is an edge from v1 to v2, then
    there must be an edge from w1 to w2"""

    Cl = vectorize_adjecency_matrix(A)
    Cr = vectorize_adjecency_matrix_2(B)
    C = Cl - Cr

    return C


def construct_1_constraints(A, B):

    n = len(A)
    C1 = construct_well_definedness_constraint(n)
    C2 = construct_bijectivity_constraint(n)
    C3 = construct_edge_preserving_constraint(A, B)

    return np.vstack((C1, C2, C3))


def construct_well_definedness_constraint(n):
    """every v is mapped to exactly one w (well-definedness)"""
    C = np.zeros((n, n**2))
    for j in range(n):
        tmp = np.zeros(n)
        tmp[j] = 1
        row = np.repeat(tmp, n)
        C[j] = row

    return C


def construct_bijectivity_constraint(n):
    """every w gets assigned one v (surjective and injective)"""
    C = np.zeros((n, n**2))
    for j in range(n):
        row = np.zeros(n**2)
        row[j*n:(j+1)*n] = np.ones(n)
        C[j] = row

    return C


def vectorize_adjecency_matrix(A):
    """vectorizes the constraint defined by the adjacency matrix A"""

    n = len(A)
    C = np.zeros((n**2, n**2))
    for j in range(n):
        for k in range(n):
            row = np.zeros(n**2)
            row[k*n:(k+1)*n] = A[j]
            C[n*j+k] = row

    return C


def vectorize_adjecency_matrix_2(B):

    n = len(B)
    C = np.zeros((n**2, n**2))
    for l in range(n):
        for j in range(n):
            row = np.zeros(n**2)
            for k in range(n):
                row[l + k*n] = B[k][j]
            C[l*n+j] = row
    return C
