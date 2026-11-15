import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab

path_to_data_folder = Path(__file__).resolve().parent.parent 
path_to_train_set = path_to_data_folder/"datasets"/"train_5500_coarse.label"
path_to_test_set = path_to_data_folder/"datasets"/"TREC_10_coarse.label"

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt

# Task 11

def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx,doc in enumerate(docs):
        G = nx.Graph()

        ##################
        # your code here #
        ##################
        for word in doc:
            if word not in list(G.nodes):
                G.add_node(word, name=vocab[word])

        for i, word in enumerate(doc):
            k = min(len(doc) - i - 1, window_size)
            for j in range(i, i + k + 1):
                for l in range(i, i + k + 1):
                    if (doc[l], doc[j]) not in list(G.edges) and l != j:
                        G.add_edge(doc[l], doc[j])

        graphs.append(G)
    
    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

print("Example of graph-of-words representation of document")
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram, HadamardCode
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Task 12

# Transform networkx graphs to grakel representations
G_train = graph_from_networkx(G_train_nx, node_labels_tag="name") # your code here #
G_test = graph_from_networkx(G_test_nx, node_labels_tag="name") # your code here #

# Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(n_iter=1, normalize=False, base_graph_kernel=VertexHistogram) # your code here #

# Construct kernel matrices
K_train = gk.fit_transform(G_train) # your code here #
K_test = gk.transform(G_test) # your code here #

#Task 13

# Train an SVM classifier and make predictions

##################
# your code here #
##################

clf = SVC(kernel="precomputed")
clf.fit(np.asarray(K_train), y_train)
y_pred = clf.predict(K_test)

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred, y_test))


#Task 14

##################
# your code here #
##################

# We train here a classifier based on VertexHistogram directly
G_train = graph_from_networkx(G_train_nx, node_labels_tag="name") # Resets the generators
G_test = graph_from_networkx(G_test_nx, node_labels_tag="name")

gk_hist = VertexHistogram(normalize=False) 

# Construct kernel matrices
K_train_hist = gk_hist.fit_transform(G_train)  
K_test_hist = gk_hist.transform(G_test) 

clf_hist = SVC(kernel="precomputed")
clf_hist.fit(np.asarray(K_train_hist), y_train)
y_pred_hist = clf_hist.predict(K_test_hist)

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred_hist, y_test))

# Accuracy for WL kernel: 0.962
# Accuracy for VH kernel: 0.972
# VH kernel outperforms WL kernel by a small margin, meaning that for this specific task, a more complex classifier is not always a better choice. 


# We train here a classifier based on HadamardCode kernel 
G_train = graph_from_networkx(G_train_nx, node_labels_tag="name") # Resets the generators
G_test = graph_from_networkx(G_test_nx, node_labels_tag="name")

gk_hada = HadamardCode(normalize=False) 

# Construct kernel matrices
K_train_hada = gk_hada.fit_transform(G_train)  
K_test_hada = gk_hada.transform(G_test) 

clf_hada = SVC(kernel="precomputed")
clf_hada.fit(np.asarray(K_train_hada), y_train)
y_pred_hada = clf_hada.predict(np.asarray(K_test_hada))

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred_hada, y_test))

# Accuracy for WL kernel: 0.962
# Accuracy for HadamardCode kernel: can't be computed
# Compting Hadamard Code kernel takes much more memory as it computes!
# It leads to a killed process on my laptop. 
# Indeed, it computes a Hadamard Matrix of size N = "the number of distinct label in the graph"
# It represents N**2 entries which is enormous... 
# We see here that computing feasability matters even more than achieving better score. 