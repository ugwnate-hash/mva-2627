"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2026
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    # your code here #
    ##################
    X_train, y_train = list(), list()
    for _ in range(n_train):
        k = np.random.randint(low=1, high=max_train_card+1)
        rand_list = list(np.random.randint(low=1, high=11, size=(k)))
        y_train.append(sum(rand_list))
        while len(rand_list) != max_train_card:
            rand_list = [0] + rand_list
        X_train.append(rand_list)
    return np.array(X_train), np.array(y_train)


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    # your code here #
    ##################
    n_test_total = 200000
    n_test = 10000
    m_max = 100
    m_min = 5
    X_test, y_test = list(), list()
    for j in range(m_min, m_max+1, 5):
        card_list = list()
        y_card = list()
        for _ in range(n_test):
            rand_list = list(np.random.randint(low=1, high=11, size=(j)))
            y_card.append(sum(rand_list))
            card_list.append(rand_list)
        y_test.append(y_card)
        X_test.append(np.array(card_list))

    return X_test, y_test

print(len(create_test_dataset()[0][1]))
print(len((create_train_dataset()[1])))