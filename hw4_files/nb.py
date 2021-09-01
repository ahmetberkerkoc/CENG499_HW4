import numpy as np
import time
from math import log
import copy
from operator import itemgetter
def vocabulary(data):
    
    set_of_word=set()
    for sentence in data:
        for word in sentence:
            if not word in set_of_word:
               set_of_word.add(word)
            
    return set_of_word
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    

def estimate_pi(train_labels):
    label = list(set(train_labels))
    pi = {}
    length = len(train_labels)

    for element in label:
        pi[element]=0
        for classes in train_labels:
            if classes == element:
                pi[element]=pi[element]+1
        pi[element]=pi[element]/length    
    
    return pi
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    
    
def estimate_theta(train_data, train_labels, vocab):
    
    set_label = list(set(train_labels))
    sentence =[]
    theta_class = {}
    theta_word = {}
    
    for classes in set_label:
        i=0
        for data in train_data:
            #print(train_labels[i]) #for debugging
            if train_labels[i] == classes:
                sentence.extend(data)
            i=i+1
        summ_down=len(sentence)
        for all_word in vocab:
            summ_up = sentence.count(all_word)
            theta_word[all_word]= (summ_up + 1) /(summ_down +len(vocab))
            
        theta_class[classes]=copy.deepcopy(theta_word)
        sentence =[]
    return theta_class
                 
    
#def flatten(t):
#    return [item for sublist in t for item in sublist]
    

def test(theta, pi, vocab, test_data):
    classes =[]
    score =[]
    score_table = []
    for class_name, prop in pi.items():
        classes.append(class_name)
    
    for data in test_data:
        score=[]
        for c in classes:
            
            sum=0
            for v in vocab:
                count = data.count(v)
                sum+=log(theta[c][v])*count
            sum+=log(pi[c])
            score1=(sum,c)
            score.append(score1)
        score_table.append(score)
    return score_table

   
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """

def calculate_accuracy(pred, truth):
    acc=0
    for i in range(len(truth)):
        if(pred[i]==truth[i]):
            acc=acc+1
            
    return 100*acc/len(truth)        
    



if __name__ == '__main__':
    start =time.time()
    train_data =[] 
    test_data = []
    test_labels=[]
    train_labels=[]
    
    filepath = 'hw4_data/sentiment/train_data.txt'
    with open(filepath,encoding='utf8') as fp:
        line = fp.readline()
        while line:
            x=line.split()
            #del x[0]
            train_data.append(x)
            #print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
    
    filepath = 'hw4_data/sentiment/train_labels.txt'
    with open(filepath,encoding='utf8') as fp:
        line = fp.readline()
        while line:
            x=line.split("\n")
            
            train_labels.append(x[0])
            #print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
    
    filepath = 'hw4_data/sentiment/test_data.txt'
    with open(filepath,encoding='utf8') as fp:
        line = fp.readline()
        while line:
            x=line.split()
            #del x[0]
            test_data.append(x)
            #print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            
    filepath = 'hw4_data/sentiment/test_labels.txt'
    with open(filepath,encoding='utf8') as fp:
        line = fp.readline()
        while line:
            x=line.split("\n")

            test_labels.append(x[0])
            #print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            
    vocab = vocabulary(train_data)
    pi=estimate_pi(train_labels)
    theta = estimate_theta(train_data,train_labels,vocab)
    table = test(theta,pi,vocab,test_data)
    predict=[]
    
    for choose in table:
        p = max(choose, key = itemgetter(0))[1]
        predict.append(p)
    ACC =calculate_accuracy(predict,test_labels)
    print("test accuracy = {}".format(ACC))
    end=time.time()
    
    print("total time for calculation = {} minutes {} second".format(int((end-start)/60),(end-start)%60))
    
    """        
    print(len(test_labels))
    print(len(train_labels))
    print(len(test_data))
    print(len(train_data))
    
    """
    
    
    
    