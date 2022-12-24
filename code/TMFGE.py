import random
import numpy as np
import sklearn
import linkPred
import time
import pickle
from classify import Classifier, read_node_label
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import networkx as nx
G=[]
Hash = {}
time_re = 0
mint = -1
edge_size = 0
dimension = 128
g = nx.MultiDiGraph()
class edge:
    def __init__(self, y, t):
        self.y = y
        self.t = t
def load_graph(file):
    global mint
    global time_re
    global edge_size
    maxt = 0
    lenth = 0
    edges = []
    f=open(file,'r')
    for l in f:
        edge_size+=1
        x,y,t = l.strip().split(",")
        x = int(x)
        y = int(y)
        if x == y:
            continue
        max_xy = max(x,y)
        if max_xy >= lenth:
            for i in range(0,max_xy-lenth+1):
                G.append([])
            lenth = len(G)
        t = int(t)
        if mint < 0 :
            mint = t
        mint = min(t,mint)
        maxt = max(maxt,t)
        edges.append((x,y,{'time': t}))
        G[x].append(edge(y,t))
        G[y].append(edge(x,t))
    f.close()
    time_re = maxt-mint
    g.add_edges_from(edges)
    f=open(file,'r')
    size = 0
    size_k = 0
    mint_now = mint
    for l in f:
        x,y,t = l.strip().split(",")
        t = int(t)
        size+=1
        if size >= edge_size/500:
            size=0
            size_k+=1
        Hash[t-mint] = size_k
    f.close()
    print("edge_size="+str(edge_size))
    print("time split number:"+str(size_k))
    return size_k,edge_size
    
def generateSequence(startIndex, path_length, alpha):
    result = [edge(startIndex,0)]
    current = startIndex
    for i in range(0, path_length):
        if random.random() < alpha:
            nextIndex = edge(result[len(result)-1].y,0)
        else:
            probs = random.randint(0,len(G[current])-1)
            nextIndex = G[current][probs]
            #print(nextIndex)
        result.append(nextIndex)

        current = nextIndex.y

    return result

def random_walk(num_paths, path_length, alpha, k):
    global mint
    global time_re
    print("max_time-min_time is:"+str(time_re))
    print(len(G))
    matrix_random = np.zeros([len(G),k+1])
    tot = 0
    tots = 0
    for i in range(1, len(G)):
        tot+=1
        if len(G[i]) != 0:
            for j in range(0, num_paths):
                indexList=generateSequence(i, path_length, alpha)
                for tmp in indexList:
                    if tmp.t != 0:
                        matrix_random[i][Hash[tmp.t-mint]]+=1
        if tot>=5000:
            tot=0
            tots+=1
            print("5000 nodes have been sampled "+str(tots)+" times")
    return matrix_random

def Proportion(A):
    row_sum = A.sum(axis = 1)
    for i in range(0,len(A)):
        if row_sum[i]!=0:
            A[i] /= row_sum[i]
    return A

def Matrix_Based(k,edge_sizes,T_lenth):
    Degree = np.zeros([len(G)-1, len(G)-1])
    A_m = np.zeros([len(G)-1, len(G)-1])
    E_m = np.zeros([len(G)-1,k+1])
    for i in range(1,len(G)):
        if len(G[i]) != 0:
            Degree[i][i] = 1. / len(G[i])
        for j in G[i]:
            A_m[i-1][j.y-1] = 1
            E_m[i-1][Hash[j.t-mint]]+=1
    E_m = Proportion(E_m)
    matrix_r = E_m
    print("Proportion Done")
    '''
    S_mid = np.dot(Degree,A_m)
    S_mid = Proportion(S_mid)
    S_mid2 = deepcopy(S_mid)
    Final_S = deepcopy(S_mid)
    print("MD start")
    for i in range(1,T_lenth):
        S_mid2 = S_mid2@S_mid
        Final_S += S_mid2
        #S_1 = Proportion(S_1)
        print(i)
    '''
    matrix = np.log(matrix_r*edge_sizes,where=(matrix_r!=0))
    return matrix

def Random_Walk_Based(k, edge_sizes, T_lenth):
    matrix_r = random_walk(100,T_lenth,0.10,k)
    matrix_r = Proportion(matrix_r)
    print("Random Walk Done.")
    matrix = np.log(edge_sizes * matrix_r,where=(matrix_r!=0))
    print("Matrix Done.")
    return matrix

def Manhattan(u,v):
    sumi = 0
    for i in range(0,len(u)):
        sumi+=abs(u[i]-v[i])
    return sumi

def node_classification(embeddings):
    X, Y = read_node_label("../data/forum/forum_label.txt")
    #print(X)
    clf_ratio = 0.9
    print("Training classifier using {:.2f}% nodes...".format(clf_ratio*100))
    clf = Classifier(vectors=embeddings,clf=LogisticRegression(max_iter = 500))
    result = clf.split_train_evaluate(X, Y, clf_ratio)

def load_embedding_from_pkl(file_path):
    Embed = []
    Kong = []
    for i in range(0,128):
        Kong.append(0)
    with open('../data/forum/forum.w2v.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    for i in range(0,len(G)):
        u = embeddings.get(i)
        if u is not None:
            Embed.append(u)
        else:
            Embed.append(Kong)
    return Embed

if __name__ == "__main__":
    k,edge_sizes = load_graph("../data/forum/forum_final.txt")
    t1 = time.perf_counter()
    T_lenth = 10
    matrix = Random_Walk_Based(k,edge_sizes,T_lenth)
    U_final,s,Z = sklearn.utils.extmath.randomized_svd(matrix, dimension)
    print("SVD Done, start to save embedding.")
    Sigma = np.zeros((dimension,dimension))
    for i in range(0,dimension):
        Sigma[i][i]=np.sqrt(s[i])
    Temp = np.dot(U_final,Sigma)
    U = Temp
    t2 = time.perf_counter()
    print(t2-t1)
    linkPred.linkPred(U)
    #node_classification(U)
        
