import networkx as nx
import numpy as np
import pickle
import random
edge_set = set()
def load_data_as_graph(path='network_data/forum.txt', weight_idx=0, time_idx=2):
    '''
    Returns a networkx graph.
    Edge property is called 'time'
    :param path: path the to dataset with header u, v, weight(u, v), time(u, v)
    :return: G
    '''
    edges = []
    with open(path) as f:
        for line in f:
            tokens = line.strip().split(",")
            u = int(tokens[0])
            v = int(tokens[1])
            time = int(int(tokens[time_idx])/100)
            #time = 1
            if weight_idx:
                weight = int(tokens[weight_idx])

                # add edge
                edges.append((u, v, {'weight': weight, 'time':time}))
            else:
                edges.append((u, v, {'time': time}))
            edge_set.add(str(u)+str(v))
    g = nx.MultiDiGraph()
    g.add_edges_from(edges)
    #print(g.edges(2, data=True))

    return g



'''def create_embedding_and_training_data_old(g, train_edges_fraction=0.1):
    edges = sorted(g.edges(data=True), key=lambda x: x[2]['time'])
    num_edges = len(edges)

    # training edges
    num_train_edges = int(train_edges_fraction * num_edges)
    train_edges = edges[:num_train_edges]

    # link prediction positive edges
    pos_edges = edges[num_train_edges:]
    neg_edges = []
    for i in range(len(pos_edges)):
        n_edge = get_negative_edge(g)
        neg_edges.append(n_edge)

    return train_edges, pos_edges, neg_edges
'''

def create_embedding_and_training_data(g, train_edges_fraction=0.1):
    '''
    Create partition of edges into
     -- embedding edges used for learning the embedding
     -- pos edges : positive example of edges for link prediction task

    :param g: nx graph
    :param train_edges_fraction: what fraction of edges to use for embedding learning
    '''

    nodes = g.nodes()
    train_edges = []
    pos_edges = []
    neg_edges = []
    edges = []
    Degree = np.zeros((g.number_of_nodes()+1,), dtype = np.int32)
    for node in nodes:    
        for e in g.edges(node, data=True):
            edges.append(e)
            #print(e[1])
            #print("Degree is " , len(Degree))
            Degree[int(e[0])]+=1
            Degree[int(e[1])]+=1
    edges = sorted(edges, key=lambda x: x[2]['time'], reverse=True)
    num_edges = len(edges)
    num_train_edges = int((1-train_edges_fraction) * num_edges)
    #time_split = edges[num_train_edges][2]['time']
    for e in edges:
        if Degree[e[0]]>1 and Degree[e[1]]>1 and num_train_edges>0:
            pos_edges.append(e)
            num_train_edges -= 1
            Degree[e[0]]-=1
            Degree[e[1]]-=1
        else:
            train_edges.append(e)

    '''for node in nodes:
        edges_of_node = []
        for e in g.edges(node, data=True): # only gets outgoing edges
            edges_of_node.append(e)
        edges_of_node = sorted(edges_of_node, key=lambda x: x[2]['time'])
        num_edges = len(edges_of_node)

            # training edges per node
        num_train_edges = int(train_edges_fraction * num_edges)
        train_edges.extend(edges_of_node[:num_train_edges+1])

            # link prediction positive edges
        pos_edges.extend(edges_of_node[max(num_train_edges-1,0):])'''
    train_edges = sorted(train_edges, key=lambda x: x[2]['time'])
    print(len(train_edges))
    print(len(pos_edges))
    nodes_size = g.number_of_nodes()
    for i in range(len(pos_edges)):
        ok = 1
        #size_epoch += 1
        while ok:
            first_node = random.randint(1,nodes_size)  # pick a random node
            second_node = random.randint(1,nodes_size)
            if first_node==second_node or str(first_node)+str(second_node) in edge_set:
                continue
            ok = 0
        n_edge = (first_node, second_node, {'weight':1, 'time': None})
        #if size_epoch >= 200:
            #size_epoch = 0
            #print("size_epoch_once")
        neg_edges.append(n_edge)

    return train_edges, pos_edges, neg_edges


def main():
    path = '../data/forum/forum.txt'
    contact_g =  load_data_as_graph(path=path, weight_idx=0, time_idx=2)
    embedding_edges, pos_edges, neg_edges = create_embedding_and_training_data(contact_g, train_edges_fraction=0.75)

    save_path = '../data/forum/forum'
    with open(save_path + '_final.txt', 'w') as f:
        for u,v,t in embedding_edges:
            f.write(str(u)+","+str(v)+","+str(t['time'])+'\n')
    with open(save_path + '_embedding_edges', 'wb') as f:
        pickle.dump(embedding_edges, f)
    with open(save_path + '_pos_edges', 'wb') as f:
        pickle.dump(pos_edges, f)
    with open(save_path + '_neg_edges', 'wb') as f:
        pickle.dump(neg_edges, f)

if __name__ == '__main__':
    main()
