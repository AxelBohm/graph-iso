import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from graphimport import *
import scipy.special
from collections import Counter


def weisfeiler_lehman(G, H, iterations=3, labels=None):
    """
    some parts of this function come from
    https://github.com/emanuele/jstsp2015
    """
    if G.number_of_nodes() != H.number_of_nodes():
        print('not of same size')
        return

    graph_list = [G, H]
    lists = []
    k = [0] * (iterations + 1)

    for g in graph_list:
        lists.append( g.adj )

    if labels is None:
        labels = [0] * 2
        for i in range(2):
            labels[i] = dict(graph_list[i].degree())

    for i in range(2):
        it = 0
        label_lookup = {}
        label_counter = 0
        new_labels = copy.deepcopy(labels)

        while it < iterations:
            label_lookup = {}
            label_counter = 0

            for v in range(len(lists[i])):
                # form a multiset label of the node v of the i'th graph
                # and convert it to a string
                adjlabel = [labels[i][k] for k in np.array(lists[i][v])]
                long_label = np.concatenate([np.array([labels[i][v]]), np.sort(adjlabel)])
                long_label_string = str(long_label)

                # if the multiset label has not yet occurred, add it to the
                # lookup table and assign a number to it
                if not (long_label_string in label_lookup):
                    label_lookup[long_label_string] = label_counter
                    new_labels[i][v] = label_counter
                    label_counter += 1
                else:
                    new_labels[i][v] = label_lookup[long_label_string]

            labels = copy.deepcopy(new_labels)
            it = it + 1

    return labels


def partialiso(v,w,g):
    k = len(v)
    for i in range(k):
        for j in range(i+1,k):
            if v[i] is v[j] and w[i] is not w[j]:
                return False
            if w[i] is w[j] and v[i] is not v[j]:
                return False
            if (not g.has_edge(v[i],v[j]) is g.has_edge(w[i],w[j])):
                return False
    return True


def kgraph(g,k):
    """
    construct k-graph to be colored
    """
    if k==1: return g
    n_nodes = g.number_of_nodes()
    kg = nx.Graph()
    for n in range(pow(n_nodes,k)):
        kg.add_node(n)
        node = multiindexmap(n,k,n_nodes)
        for d in range(k):
            o_node = node.copy()
            for j in range(n_nodes):
                o_node[d] = j
                o_index = indexmap(o_node,n_nodes)
                if o_index < n:
                    kg.add_edge(n,o_index)
    return kg


def initkcolor(g,k):
    """
    initial coloring for a k-graph
    """
    if k==1: return dict(g.degree())
    n_nodes = g.number_of_nodes()
    label = [None] * pow(n_nodes,k)
    label_counter = 0

    while None in label:
        n = label.index(None)
        node = multiindexmap(n, k, n_nodes)
        for on in range(n):
            onode = multiindexmap(on, k, n_nodes)
            if partialiso(node, onode, g):
                label[n] = label[on]
                break
        if label[n] is None:
            label[n] = label_counter
            label_counter += 1
    return label


def indexmap(multiindex, n_nodes):
    """
    return bj from multiindex of length d
    and order n to range [0,n^d]
    inverse is multiindexmap

    >>> indexmap([2,1,0],5)
    7
    """
    mapto = multiindex[0]
    for d in range(1, len(multiindex)):
        mapto += multiindex[d] * pow((n_nodes), d)
    return mapto


def multiindexmap(index, k, n_nodes):
    """
    return bj from index to multiindex
    of length d and order n
    inverse is indexmap

    >>> multiindexmap(7,3,5)
    [2, 1, 0]
    """
    multiindex = [0] * k
    s = 0
    for d in range(k-1,-1,-1):
        multiindex[d] = (index-s) // pow((n_nodes),d)
        s += multiindex[d] * pow((n_nodes),d)
    return multiindex


def indexloop(func,dim,n_nodes):
    """
    compressed nested loops of arbitrary depth - deprecated
    """
    i = [0] * (dim+1) # indices (dim+1) only used as stopping criterion
    p = 0 # Used to increment all of the indicies correctly, at the end of each loop.
    while p != dim:
        index = i[0:dim]
        func(index)
        # increment all of the indicies
        i[0] += 1
        p = 0
        while i[p] == n_nodes:
            i[p]=0
            p += 1
            i[p] += 1

def test_example(k):
    """
    test two V10 graphs for different k
    >>> test_example(1)
    [[4, 6], [4, 6]]

    >>> test_example(3)
    [[2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16], [4, 6, 6, 6, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 48]]
    """
    G = file2graph("graphs/CaiOrigTwistedV10.txt")
    H = file2graph("graphs/CaiOrigV10.txt")
    labels = weisfeiler_lehman(G, H)
    kG = kgraph(G,k)
    kH = kgraph(H,k)
    colors = [initkcolor(G,k), initkcolor(H,k)]
    iterations = 4
    labels = weisfeiler_lehman(kG, kH, iterations, colors)
    keylist = []
    for l in labels:
        temp = list(l.values()) if isinstance(l,dict) else l
        keylist.append(sorted(dict(Counter(temp)).values()))
    return keylist


if __name__ == '__main__':
    import doctest
    doctest.testmod()
