import networkx as nx
import matplotlib.pyplot as plt
from weisfeiler_lehman import *

def file2graph(pathtofile):
    g = nx.Graph()
    with open(pathtofile) as f:
        dlines = f.readlines()
        wait = 1
        for l in dlines:
            if(l=="edges:\n"):
                wait=0
                continue
            if(wait==1): continue
            # split at white space
            l = l.split()
            g.add_node(int(l[0]))
            g.add_node(int(l[1]))
            g.add_edge(int(l[0]),int(l[1]))
    return g

def drawgraph(graph):
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()

if __name__ == '__main__':
    g = file2graph("graphs/CaiOrigTwistedV10.txt")
    nx.draw(g)
    plt.show()
