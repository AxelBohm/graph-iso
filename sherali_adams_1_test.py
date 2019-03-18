import networkx as nx
import numpy as np
import sherali_adams as sa
import k_one_helpers
from graphimport import file2graph
import pdb


def test_simple1():
    G = file2graph("graphs/simple1.txt")
    H = file2graph("graphs/simple2.txt")
    k = 1

    (answer, isomorphism) = sa.k_dim_linear_relax(G, H, k)
    assert answer in (0, 3)
    if answer is not 0:
        np.testing.assert_almost_equal(isomorphism.sum(0), 1)
        np.testing.assert_almost_equal(isomorphism.sum(1), 1)


def test_if_solution_is_bijection():
    """after converting the solution back to a square matrix all row sums and
    all colums sums should be 1"""

    n = 4
    k = 1
    G = nx.path_graph(n)
    H = nx.path_graph(n)

    (answer, X) = sa.k_dim_linear_relax(G, H, k)

    np.testing.assert_almost_equal(X.sum(0), np.ones(n))
    np.testing.assert_almost_equal(X.sum(1), np.ones(n))


def test_if_answer_is_correct():
    n = 5
    G = nx.path_graph(n)
    H = nx.path_graph(n)
    k = 1

    (answer, X) = sa.k_dim_linear_relax(G, H, k)

    assert answer


def test_if_answer_is_correct2():
    """ test if different number of nodes is immediately realized"""
    G = nx.path_graph(4)
    H = nx.path_graph(5)
    k = 1

    (answer, X) = sa.k_dim_linear_relax(G, H, k)

    assert not(answer)


def test_first_row_B():
    """ test first row for the case of a path graph of length 4 """

    G = nx.path_graph(4)
    B = nx.adjacency_matrix(G)
    B = B.toarray()
    first_row = k_one_helpers.vectorize_adjecency_matrix_2(B)[0]
    supposed_to_be_first_row = np.zeros(4**2)
    supposed_to_be_first_row[4] = 1
    assert np.all(first_row == supposed_to_be_first_row)


def test_well_definedness_length():
    """ should result in n constraints"""
    n = 5
    C = k_one_helpers.construct_well_definedness_constraint(n)
    assert C.shape[0] == n


def test_bijectivity_length():
    """ should result in n constraints"""
    n = 5
    C = k_one_helpers.construct_well_definedness_constraint(n)
    assert C.shape[0] == n


def test_well_definedness_rowsum():
    n = 5
    C = k_one_helpers.construct_well_definedness_constraint(n)
    assert np.all(C.sum(axis=1) == n*np.ones(n))


def test_well_definedness_colsum():
    n = 5
    C = k_one_helpers.construct_well_definedness_constraint(n)
    assert np.all(C.sum(axis=0) == np.ones(n**2))
