## Wrapper for online LDA

import numpy as np
import onlineldavb

def initialize_onlineLDA(vocab, K, D, tau0=1024, kappa=.7):
    """
    vocab: Set of known words. Words outside this list are ignored
    K: Number of topics
    D: Total number of expected documents (high)
    tau0, kappa: Parameters for the online gradient descent (do not touch!)
    """
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, tau0, kappa)
    return olda


def train_onlineLDA(olda, trainset):
    """
    trainset: Set of documents used for unsupervised training of LDA
    """
    N = len(trainset)
    for i in range(N):
        (gamma_m, bound_m) = olda.update_lambda_docs([trainset[i]])
        if(i%100==0):
            print(int(i / N * 100), "%")


def get_topic_word_distribution(olda):
    lambdas = olda._lambda
    row_sums = lambdas.sum(axis=1)
    return lambdas / row_sums[:, np.newaxis]


def get_documents_topic(olda, doc):
    """
    doc: Document
    """
    gammas, _ = olda.do_e_step_docs([doc])
    gamma = gammas[0] / sum(gammas[0])
    return gamma


def get_document_perplexity(olda, D, doc):
    """
    doc: Document to analyze
    Output: (Logikelihood bound, perplexity)
    """
    (wordids, wordcts) = onlineldavb.parse_doc_list([doc], olda._vocab)
    gammas, _ = olda.do_e_step_docs([doc])
    bound = olda.approx_bound_docs([doc], gammas)
    s = sum(map(sum, wordcts))
    
    if(s == 0):
        return None, None
    
    perwordbound = bound * 1.0 / (D * s)
    return (perwordbound, np.exp(-perwordbound))


def get_average_perplexity(olda, D, docs):
    """
    Compute the average perplexity of the set of documents "docs"
    """
    sum = 0
    for doc in docs :
        temp = get_document_perplexity(olda, D, doc)[1]
        n = len(docs)
        if (temp != None) :
            sum += temp
            n -= 1
    return sum/n


def filtering(p, size=25):
    """
    Smooth a curve
    """
    
    r = []
    for i in range(size-1, len(p)-size+1):
        r += [sum(p[i-(size-1):i+size])/size]
    return r


def visualize_topics(olda, t) :
    """
    Returns a list of K topics represented by t top words
    """
    
    topic_distribution = get_topic_word_distribution(olda)
    reversed_vocab = dict( (v, k) for k, v in olda._vocab.items() )
    n = len(topic_distribution)
    topic_index = [[] for k in range(n)]
    topic_list = [[] for k in range(n)]
    
    for k in range(n) :
        temp = list(topic_distribution[k])
        for i in range(t):
            topic_index[k].append(np.argmax(temp))
            topic_list[k].append(reversed_vocab.get(topic_index[k][i], "error##"))
            temp.remove(max(temp))
    
    return topic_list