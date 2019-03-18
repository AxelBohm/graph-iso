from sherali_adams import k_dim_linear_relax
import networkx as nx
from graphimport import file2graph
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction


def example():
    G = file2graph("graphs/CaiOrigTwistedV10.txt")
    H = file2graph("graphs/CaiOrigV10.txt")
    nx.draw(G)
    print('close figure to continue')
    plt.show()
    nx.draw(H)
    print('close figure to continue')
    plt.show()
    for k in range(1, 4):
        print("\033[1;33;40m \n running Sherali-Adams with k =", k, "\033[0m ")
        solver = "linear" if k < 3 else "minimize"
        (answer, isomorphism) = k_dim_linear_relax(G, H, k, solver)
        print("\033[01;32m \n", flag_interpreter(answer), "\033[00m ")
        print("\033[1;33;40m \n", np.round(isomorphism, decimals=5),
              "\033[0m ")


def flag_interpreter(flag):

    if flag == 0:
        answer = "graphs are definitely not isomorphic"
    elif flag == 1:
        answer = "The two graphs are isomorphic"
    elif flag == 2:
        answer = "isomorphism up to numerical errors"
    elif flag == 3:
        answer = "fractional isomorphism found"
    else:
        ValueError("not defined")

    return answer


if __name__ == '__main__':
    example()
