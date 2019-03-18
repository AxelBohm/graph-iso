# k-WL-relax
k-dimensional Weisfeiler-Lehman and its Sherali-Adams relaxation

## Importing graphs
Importing Graphs from a text file containing the edges works with
~~~~
from graphimport import file2graph
G = file2graph("graphs/CaiOrigTwistedV10.txt")
H = file2graph("graphs/CaiOrigV10.txt")
~~~~


## k-Weisfeiler-Lehman
in python we can import the k-WL algorithm using
~~~~
from weisfeiler_lehman import *
~~~~
To run Weisfeiler Lehman (with k=1) we can simply pass the two graphs to
~~~~
iterations = 3
labels = weisfeiler_lehman(G, H, iterations)
~~~~
The number of iterations can be passed to weisfeiler_lehman in the third argument.
The returned list contains two elements, for graph G and H respectivels. 
label[0] will be the (key,value) pairs of (vertex, vertexcolor) for graph G, similar for labels[1] for graph H.

We can draw the graphs with the nodes labeled by their color with
~~~~
nx.draw(G,labels=labels[0])
plt.show()
nx.draw(H,labels=labels[1])
plt.show()
~~~~

To run k-WL with k>1 we first need to construct the k-graph using
~~~~
k=3
kG = kgraph(G,k)
kH = kgraph(H,k)
~~~~

We can then set up the initial coloring on the k-graph with
~~~~
colors = [0] * 2
colors[0] = initkcolor(G,k)
colors[1] = initkcolor(H,k)
~~~~

We run the same algorithm, now on the k-graphs, passing our new initial coloring as the 4-th argument:
~~~~
iterations = 4
labels = weisfeiler_lehman(kG, kH, iterations, colors)
~~~~

To count the number of different colors in the output we can use 
~~~~
from weisfeiler_lehman_example import compare_examples
compare_examples(labels)
~~~~

For an example run
~~~~
python weisfeiler_lehman_example.py
~~~~
(or python3).

## Sherali-Adams
in python we can import the Sherali-Adams relaxation to solve the graph isomorphism problem via
~~~~
from sherali_adams import k_dim_linear_relax
~~~~
To run the Sherali-Adams relaxtion (with arbitrary k) we can simply pass the two graphs to
~~~~
k = 2
solver = "minimize"
(answer, isomorphism) = k_dim_linear_relax(G, H, k, solver)
~~~~
where `solver` can be either 
* _linear_ for a LP-solver (slow, but tends to give integer solutions),
* _minimize_ for L-BFGS solving <img src="https://latex.codecogs.com/gif.latex?\min\|Cx-b\|^2" /> instead of solving <img src="https://latex.codecogs.com/gif.latex?Cx=b" />.
The `answer` flag encodes
~~~~
if flag == 0:
    answer = "graphs are definitely not isomorphic"
elif flag == 1:
    answer = "The two graphs are isomorphic"
elif flag == 2:
    answer = "isomorphism up to numerical errors"
elif flag == 3:
    answer = "fractional isomorphism found"
~~~~

For an example run
~~~~
python sherali_adams_example.py
~~~~
(or python3).
