from __future__ import division

import math
import os
import nltk
import collections
import numpy as np
import re
import random

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from collections import defaultdict
from bs4 import BeautifulSoup


# Path to dataset
PATH_TO_DATA = "/home/bita/programming/NaiveBayes/hw2/large_movie_review_dataset" # FILL IN THE ABSOLUTE PATH TO THE DATASET HERE
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")


def tokenize_doc(doc):

    # clean = re.compile('<.*?>')
    # doc=re.sub(clean, '', doc)

    dataset = nltk.sent_tokenize(doc)

    stopwords = set(nltk.corpus.stopwords.words('english'))
    bow = defaultdict(float)

    for i in range(len(dataset)):
        dataset[i] = dataset[i].lower()
        dataset[i] = re.sub(r'\W', ' ', dataset[i])  # Remove all non-word characters
        dataset[i] = re.sub(r'\s+', ' ', dataset[i])  # Remove all punctuations.

        words = nltk.word_tokenize(dataset[i])
        # words=[w for w in words if w not in stopwords]

        for word in words:
            bow[word] += 1

    return bow

class NaiveBayes:

    def __init__(self):
        self.vocab = set()

        self.class_total_doc_counts ={}
        self.class_total_word_counts = {}
        self.class_word_counts = {}

    def train_model(self, num_docs=None):

        labels_list=defaultdict()

        for label in os.listdir(TRAIN_DIR) :
            labels_list[label]=os.path.join(TRAIN_DIR,label)

        # print ("Starting training with paths %s and %s" % (label, path)for (label, path) in labels_list.items())
        for (label, path) in labels_list.items():
            filenames = os.listdir(path)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(path,f),'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        print ("REPORTING CORPUS STATISTICS")
        for label,num_of_docs in self.class_total_doc_counts.items():
            print ("NUMBER OF DOCUMENTS IN ",label.upper()," CLASS:", num_of_docs)

        for label,num_of_tokens in self.class_total_word_counts.items():
            print ("NUMBER OF TOKENS IN ",label.upper()," CLASS:", num_of_tokens)

        print ("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))

    def update_model(self, bow, label):

        for word, count in bow.items():
            self.vocab.add(word)
            if label not in self.class_word_counts.keys():
                self.class_word_counts.update({ label:defaultdict(float) })
            if word not in self.class_word_counts[label].keys():
                self.class_word_counts[label].update({word:count})
            else:
                self.class_word_counts[label][word] = self.class_word_counts[label][word] + count

        if label not in self.class_total_doc_counts.keys():
            self.class_total_doc_counts.update({label :1})
            self.class_total_word_counts.update({label : sum(bow.values())})
        else:
            self.class_total_doc_counts[label] = self.class_total_doc_counts[label] + 1
            self.class_total_word_counts[label] = self.class_total_word_counts[label] + sum(bow.values())

    def tokenize_and_update_model(self, doc, label):


        bow = tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):

        counter = collections.Counter(self.class_word_counts[label])
        most_frequent = defaultdict(float)

        for word, count in counter.most_common(n):
            most_frequent[word] = count

        return most_frequent.items()

    def p_word_given_label(self, word, label):

        return self.class_word_counts[label][word]/self.class_total_word_counts[label]

    def p_word_given_label_and_psuedocount(self, word, label, alpha):

        return (self.class_word_counts[label][word]+alpha)/(self.class_total_word_counts[label]+(len(self.vocab)*alpha))

    def log_likelihood(self, bow, label, alpha):

        ln_likelihood = 0.0
        for word in bow.keys():
            ln_likelihood += math.log(self.p_word_given_label_and_psuedocount(word, label, alpha))

        return ln_likelihood

    def log_prior(self, label):
        return math.log(self.class_total_doc_counts[label]/sum(self.class_total_doc_counts.values()))

    def unnormalized_log_posterior(self, bow, label, alpha):
        return self.log_likelihood(bow,label,alpha)+self.log_prior(label)

    def classify(self, bow, alpha):

        labels_posteriori=defaultdict(int)
        for label in self.class_word_counts.keys():
            labels_posteriori[label]=self.unnormalized_log_posterior(bow,label,alpha)

        max_posteriori = max(labels_posteriori.items(), key=lambda x: x[1])         #return label with maximum value

        return max_posteriori[0]

    def evaluate_classifier_accuracy(self, alpha):

        classified_results=defaultdict(float)

        labels_list = defaultdict(int)

        for label in os.listdir(TEST_DIR):
            labels_list[label] = os.path.join(TEST_DIR, label)

        length=0
        for (label, p) in labels_list.items():
            filenames = os.listdir(p)
            length+=len(filenames)
            for f in filenames:
                with open(os.path.join(p, f), 'r') as doc:
                    content = doc.read()
                    bow = tokenize_doc(content)
                    classified_label = self.classify(bow, alpha)

                    #if correctly classified add it to results
                    if label == classified_label:
                        classified_results[label]+=1

        return sum(classified_results.values())/length

def produce_hw2_results():

    print ("VOCABULARY SIZE: " + str(len(nb.vocab)))
    print ()

    for label in nb.class_total_doc_counts.keys():
        print ("TOP 10 WORDS FOR CLASS " + label.upper() + " :")
        for tok, count in nb.top_n(label, 10):
            print (tok, ' : ', count)
        print()
    print()


    alpha = 1
    total_accuracy = nb.evaluate_classifier_accuracy(alpha)
    print("Total accuracy with alpha =", alpha, "is : ", total_accuracy)
    print()

    print('[done.]')

if __name__ == '__main__':
    nb = NaiveBayes()
    nb.train_model()
    produce_hw2_results()