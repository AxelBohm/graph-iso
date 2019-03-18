from weisfeiler_lehman import *

def compare_examples(labels):
    keylist = []
    for l in labels:
        temp = list(l.values()) if isinstance(l,dict) else l
        keylist.append(sorted(dict(Counter(temp)).values()))
    if (keylist[0]==keylist[1]):
        print('The number of different colors is the same - cannot distinguish')
    else:
        print('Graphs are not isomorphic!')
    answer = input("Print number of different color labels? [Y,n]")
    if answer == "n":
        return
    else:
        print(sorted(keylist[0]),sorted(keylist[1]))


def example():
    G = file2graph("graphs/CaiOrigTwistedV10.txt")
    H = file2graph("graphs/CaiOrigV10.txt")
    labels = weisfeiler_lehman(G, H)
    nx.draw(G,labels=labels[0])
    print('close figure to continue')
    plt.show()
    nx.draw(H,labels=labels[1])
    print('close figure to continue')
    plt.show()
    for k in range(1,4):
        print("\033[1;33;40m \n running k-WL with k =", k,"\033[0m ")
        kG = kgraph(G,k)
        kH = kgraph(H,k)
        colors = [0] * 2
        colors[0] = initkcolor(G,k)
        colors[1] = initkcolor(H,k)
        iterations = 4
        labels = weisfeiler_lehman(kG, kH, iterations, colors)
        compare_examples(labels)


def example2():
    G = file2graph("graphs/GroheBookFig3.4origV40.txt")
    H = file2graph("graphs/GroheBookFig3.4twistedV40.txt")
    labels = weisfeiler_lehman(G, H)
    nx.draw(G,labels=labels[0])
    print('close figure to continue')
    plt.show()
    nx.draw(H,labels=labels[1])
    print('close figure to continue')
    plt.show()
    for k in range(1,4):
        print("\033[1;33;40m \n running k-WL with k =", k,"\033[0m ")
        kG = kgraph(G,k)
        kH = kgraph(H,k)
        colors = [0] * 2
        colors[0] = initkcolor(G,k)
        colors[1] = initkcolor(H,k)
        iterations = 4
        labels = weisfeiler_lehman(kG, kH, iterations, colors)
        compare_examples(labels)


if __name__ == '__main__':
    example()
    example2()
