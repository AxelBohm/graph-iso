import networkx as nx
import numpy as np
import sherali_adams as sa
from graphimport import file2graph
import pdb


def test_simple1():
    G = file2graph("graphs/simple1.txt")
    F = file2graph("graphs/simple2.txt")
    k = 1
    (answer, isomorphism) = sa.k_dim_linear_relax(G, F, k, solver='linear')
    assert answer in (0, 3)
    if answer is not 0:
        np.testing.assert_almost_equal(isomorphism.sum(0), 1)
        np.testing.assert_almost_equal(isomorphism.sum(1), 1)


def test_simple():
    G = file2graph("graphs/simple1.txt")
    F = file2graph("graphs/simple2.txt")
    k = 2
    (answer, isomorphism) = sa.k_dim_linear_relax(G, F, k, solver='linear')
    assert answer in (0, 3)
    if answer is not 0:
        np.testing.assert_almost_equal(isomorphism.sum(0), 1)
        np.testing.assert_almost_equal(isomorphism.sum(1), 1)


def test_simple2():
    G = file2graph("graphs/simple1.txt")
    F = file2graph("graphs/simple2.txt")
    k = 3
    (answer, isomorphism) = sa.k_dim_linear_relax(G, F, k, solver='linear')
    assert answer in (0, 3)
    if answer is not 0:
        np.testing.assert_almost_equal(isomorphism.sum(0), 1)
        np.testing.assert_almost_equal(isomorphism.sum(1), 1)


def test_k_relax():
    n = 2
    k = 2
    G = nx.path_graph(n)
    H = nx.path_graph(n)

    (answer, isomorphism) = sa.k_dim_linear_relax(G, H, k)

    np.testing.assert_almost_equal(isomorphism.sum(0), np.ones(n))
    np.testing.assert_almost_equal(isomorphism.sum(1), np.ones(n))
    assert answer != 0


def test_k_relax3():
    n = 2
    k = 3
    G = nx.path_graph(n)
    H = nx.path_graph(n)

    (answer, isomorphism) = sa.k_dim_linear_relax(G, H, k)

    np.testing.assert_almost_equal(isomorphism.sum(0), np.ones(n))
    np.testing.assert_almost_equal(isomorphism.sum(1), np.ones(n))
    assert answer != 0


def test_relax():
    n = 2
    k = 3
    G = nx.path_graph(n)
    H = nx.path_graph(n)

    (answer, isomorphism) = sa.k_dim_linear_relax(G, H, k)

    np.testing.assert_almost_equal(isomorphism.sum(0), np.ones(n))
    np.testing.assert_almost_equal(isomorphism.sum(1), np.ones(n))
    assert answer != 0


def test_fancy_graph():
    G = file2graph("graphs/CaiOrigTwistedV10.txt")
    F = file2graph("graphs/CaiOrigV10.txt")
    k = 2
    (answer, isomorphism) = sa.k_dim_linear_relax(G, F, k)
    assert answer in (0, 3)
    if answer is not 0:
        np.testing.assert_almost_equal(isomorphism.sum(0), 1, decimal=4)
        np.testing.assert_almost_equal(isomorphism.sum(1), 1, decimal=4)


# def test_fancy_graph3():
#     G = file2graph("graphs/CaiOrigTwistedV10.txt")
#     F = file2graph("graphs/CaiOrigV10.txt")
#     k = 3
#     (answer, isomorphism) = sa.k_dim_linear_relax(G, F, k)
#     assert answer in (0, 3)
#     if answer is not 0:
#         np.testing.assert_almost_equal(isomorphism.sum(0), 1)
#         np.testing.assert_almost_equal(isomorphism.sum(1), 1)


def test_partial_bijection():
    v = (1, 2)
    w = (1, 2)
    assert sa.is_partial_bijection(v, w)
    v = (1, 1)
    w = (1, 1)
    assert sa.is_partial_bijection(v, w)
    v = (1, 1)
    w = (2, 1)
    assert not sa.is_partial_bijection(v, w)
    v = (1, 1, 3)
    w = (1, 1, 2)
    assert sa.is_partial_bijection(v, w)
    v = (1, 2, 3)
    w = (1, 3, 2)
    assert sa.is_partial_bijection(v, w)
    v = (1, 1, 3)
    w = (1, 3, 2)
    assert not sa.is_partial_bijection(v, w)


def test_k_relax2():
    n = 3
    k = 2
    G = nx.path_graph(n)
    H = nx.path_graph(n)

    sa.k_dim_linear_relax(G, H, k)

    (answer, isomorphism) = sa.k_dim_linear_relax(G, H, k)

    np.testing.assert_almost_equal(isomorphism.sum(0), np.ones(n))
    np.testing.assert_almost_equal(isomorphism.sum(1), np.ones(n))
