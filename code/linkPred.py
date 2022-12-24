import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def load_node_embeddings(path_to_w2v):
    with open(path_to_w2v, 'r') as f:
        embeddings = f.readlines()
        #embeddings = embeddings[1:-1]
        for i in range(len(embeddings)):
            embeddings[i] = list(map(float, embeddings[i].split(' ')))
            embeddings[i] = np.array(embeddings[i][1:-1])
    #print(embeddings)
    return embeddings


def load_edges(save_path):
    with open(save_path + 'pos_edges', 'rb') as f:
        pos_edges = pickle.load(f)
    with open(save_path + 'neg_edges', 'rb') as f:
        neg_edges = pickle.load(f)

    return pos_edges, neg_edges


def operator(u, v, op='mean'):
    if op=='mean':
        return (u + v)/2.0
    elif op=='l1':
        return np.abs(u - v)
    elif op == 'l2':
        return np.abs(u - v)**2
    elif op == 'hadamard':
        return np.multiply(u,v)
    else:
        return None


def get_dataset_from_embedding(embeddings, pos_edges, neg_edges, op='mean'):
    '''
    op can take values from 'mean', 'l1', 'l2', 'hadamard'
    '''
    y = []
    X = []
    # process positive links
    print(len(embeddings))
    for u, v, prop in pos_edges:
        # get node representation and average them
        if int(u)>=len(embeddings) or int(v)>=len(embeddings):
            continue
        u_enc = embeddings[u]
        v_enc = embeddings[v]

        '''if (u_enc is None) or (v_enc is None):
            continue'''

        datapoint = operator(u_enc, v_enc, op=op)  # (u_enc + v_enc)/2.0

        X.append(datapoint)
        y.append(0.0)

    # process negative links
    for u, v, prop in neg_edges:
        # get node representation and average them
        if int(u)>=len(embeddings) or int(v)>=len(embeddings):
            continue
        u_enc = embeddings[u]
        v_enc = embeddings[v]

        '''if (u_enc is None) and (v_enc is not None):
            u_enc = v_enc
        if (v_enc is None) and (u_enc is not None):
            v_enc = u_enc
        if (u_enc is None) and (v_enc is None):
            continue'''

        datapoint = operator(u_enc, v_enc, op=op)  # (u_enc + v_enc) / 2.0

        X.append(datapoint)
        y.append(1.0)
    dataset = np.array(X), np.array(y)
    return dataset


def linkPred(embedding):
    #pathing = 'data/outcome/outcome_c2.txt'
    #embeddings_path = 'data/cacit_all1.txt'
    #embeddings = load_node_embeddings(embeddings_path)
    embeddings = embedding
    edges_save_basepath = '../data/forum/forum_'
    pos_edges, neg_edges = load_edges(edges_save_basepath)

    X, y = get_dataset_from_embedding(embeddings, pos_edges, neg_edges)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(len(X_train))
    print(len(X_test))
    print(len(y_train))
    print(len(y_test))
    logReg = LogisticRegression(max_iter = 800, n_jobs=16)
    logReg.fit(X_train, y_train)

    y_pred = logReg.predict(X_test)
    y_score = logReg.predict_proba(X_test)
    np.savetxt(edges_save_basepath+'result_pred.txt', y_score, fmt='%.04f')
    np.savetxt(edges_save_basepath+'result_pred_true.txt', y_test, fmt='%d')
    
    print('Link prediction accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    print('Link prediction roc:', metrics.roc_auc_score(y_true=y_test, y_score=y_score[:, 1]))
    print('Link prediction AP:', metrics.average_precision_score(y_true=y_test, y_score=y_score[:, 1], average='macro', pos_label=1, sample_weight=None))
    f = open(edges_save_basepath+"link_pre_result_final.txt",'w')
    f.write('Link prediction accuracy:' + str(metrics.accuracy_score(y_true=y_test, y_pred=y_pred)) + '\n')
    f.write('Link prediction roc:' + str(metrics.roc_auc_score(y_true=y_test, y_score=y_score[:, 1])) + '\n')
    f.write('Link prediction AP:' + str(metrics.average_precision_score(y_true=y_test, y_score=y_score[:, 1], average='macro', pos_label=1, sample_weight=None)))
    f.close()