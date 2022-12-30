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

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

# Path to dataset
PATH_TO_DATA = "/home/bita/programming/NaiveBayes/hw2/large_movie_review_dataset" # FILL IN THE ABSOLUTE PATH TO THE DATASET HERE
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")



# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)



def tokenize_doc(doc):
    """
    IMPLEMENT ME!

    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """

    # clean = re.compile('<.*?>')
    # doc=re.sub(clean, '', doc)

    dataset = nltk.sent_tokenize(doc)

    stopwords=set(nltk.corpus.stopwords.words('english'))
    bow=defaultdict(float)

    for i in range(len(dataset)):
        dataset[i]=dataset[i].lower()
        dataset[i] = re.sub(r'\W', ' ', dataset[i])   # Remove all non-word characters
        dataset[i] = re.sub(r'\s+', ' ', dataset[i])  # Remove all punctuations.

        # lemmatized_sentence=lemmatize_sentence(dataset[i])
        # words=nltk.word_tokenize(lemmatized_sentence)
        words=nltk.word_tokenize(dataset[i])
        # words = [w for w in words if w not in stopwords]

        for word in words:
            bow[word]+=1

    return bow

class NaiveBayes:
    """A Naive Bayes model for text classification."""
    def __init__(self):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }

    def train_model(self, num_docs=None):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.

        num_docs: set this to e.g. 10 to train on only 10 docs from each category.
        """
        if num_docs is not None:
            print ("Limiting to only %s docs per clas" % num_docs)

        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        print ("Starting training with paths %s and %s" % (pos_path, neg_path))
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print ("REPORTING CORPUS STATISTICS")
        print ("NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL])
        print ("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL])
        print ("NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL])
        print ("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL])
        print ("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))

    def update_model(self, bow, label):

        """
        IMPLEMENT ME!

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """

        for word,count in bow.items():
            self.vocab.add(word)
            self.class_word_counts[label][word] = self.class_word_counts[label][word] + count

        self.class_total_doc_counts[label] = self.class_total_doc_counts[label] + 1
        self.class_total_word_counts[label] = self.class_total_word_counts[label] + sum(bow.values())


    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """
        Implement me!

        Returns the most frequent n tokens for documents with class 'label'.
        """
        counter=collections.Counter(self.class_word_counts[label])
        most_frequent=defaultdict(float)

        for word,count in counter.most_common(n):
            most_frequent[word]=count

        return most_frequent.items()

    def p_word_given_label(self, word, label):

        """
        Implement me!

        Returns the probability of word given label (i.e., P(word|label))
        according to this NB model.
        """

        return self.class_word_counts[label][word]/self.class_total_word_counts[label]

    def p_word_given_label_and_psuedocount(self, word, label,alpha):
        """
        Implement me!

        Returns the probability of word given label wrt psuedo counts.
        alpha - psuedocount parameter
        """
        return (self.class_word_counts[label][word]+alpha)/(self.class_total_word_counts[label]+(len(self.vocab)*alpha))

    def log_likelihood(self, bow, label, alpha):

        """
        Computes the log likelihood of a set of words give a label and psuedocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; psuedocount parameter
        """
        ln_likelihood = 0.0
        for word in bow.keys():
            ln_likelihood += math.log(self.p_word_given_label_and_psuedocount(word, label, alpha))

        return ln_likelihood

    def log_prior(self, label):

        """
        Implement me!

        Returns a float representing the fraction of training documents
        that are of class 'label'.
        """
        return math.log(self.class_total_doc_counts[label]/(self.class_total_doc_counts[POS_LABEL]+self.class_total_doc_counts[NEG_LABEL]))


    def unnormalized_log_posterior(self, bow, label, alpha):

        """
        Implement me!

        alpha - psuedocount parameter
        bow - a bag of words (i.e., a tokenized document)
        Computes the unnormalized log posterior (of doc being of class 'label').
        """
        return self.log_likelihood(bow,label,alpha)+self.log_prior(label)

    def classify(self, bow, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        bow - a bag of words (i.e., a tokenized document)

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior).
        """
        pos_posteriori = self.unnormalized_log_posterior(bow,POS_LABEL,alpha)
        neg_posteriori = self.unnormalized_log_posterior(bow,NEG_LABEL,alpha)

        if pos_posteriori>neg_posteriori:
            return POS_LABEL
        elif pos_posteriori<neg_posteriori:
            return NEG_LABEL
        else:
            return random.choice([POS_LABEL,NEG_LABEL])

    def likelihood_ratio(self, word, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        return self.p_word_given_label_and_psuedocount(word, POS_LABEL, alpha) / self.p_word_given_label_and_psuedocount(word, NEG_LABEL, alpha)


    def evaluate_classifier_accuracy(self, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        pos_classified={POS_LABEL:0,NEG_LABEL:0}
        neg_classified={POS_LABEL:0,NEG_LABEL:0}


        pos_path = os.path.join(TEST_DIR, POS_LABEL)
        neg_path = os.path.join(TEST_DIR, NEG_LABEL)
        pos_length=0
        neg_length=0
        print ("Starting testing with paths %s and %s" % (pos_path, neg_path))
        print()
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            counter = 5
            length=len(filenames)

            if label ==POS_LABEL:
                pos_length=length
            else:
                neg_length=length
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    bow=tokenize_doc(content)
                    classified_label=self.classify(bow,alpha)

                    if label==POS_LABEL:
                        pos_classified[classified_label]+=1
                    else:
                        neg_classified[classified_label]+=1

                    if classified_label!=label and counter:
                        print(f," file in ",label,"class is worngly detected.")
                        counter=counter-1
            print()

        print("Number of positive missclassified examples :",pos_classified[NEG_LABEL])
        print("Number of negative missclassified examples :",neg_classified[POS_LABEL])
        print()

        return pos_classified[POS_LABEL]/pos_length, neg_classified[NEG_LABEL]/neg_length,\
               (pos_classified[POS_LABEL]+neg_classified[NEG_LABEL])/(pos_length+neg_length)


def produce_hw2_results():
    # PRELIMINARIES
    # uncomment the following 9 lines after you've implemented tokenize_doc
    d1 = "this sample doc has   words that  repeat repeat"
    bow = tokenize_doc(d1)
    alpha =1

    assert bow['this'] == 1
    assert bow['sample'] == 1
    assert bow['doc'] == 1
    assert bow['has'] == 1
    assert bow['words'] == 1
    assert bow['that'] == 1
    assert bow['repeat'] == 2
    print ('')

    # QUESTION 1.1
    # Implementation only

    # QUESTION 1.2
    # uncomment the next two lines when ready to answer question 1.2
    print("VOCABULARY SIZE: " + str(len(nb.vocab)))
    print()

    # QUESTION 1.3
    # uncomment the next set of lines when ready to answer qeuestion 1.2

    print ("TOP 10 WORDS FOR CLASS " + POS_LABEL + " :")
    for tok, count in nb.top_n(POS_LABEL, 10):
        print (tok,' : ' ,count)
    print()

    print ("TOP 10 WORDS FOR CLASS " + NEG_LABEL + " :")
    for tok, count in nb.top_n(NEG_LABEL, 10):
        print (tok,' : ' ,count)
    print ('')

    #Question 2
    print("fantastic probability given positive documents : ", nb.p_word_given_label("fantastic", POS_LABEL))
    print("fantastic probability given negative documents : ", nb.p_word_given_label("fantastic", NEG_LABEL))
    print()
    print("boring probability given negative documents : ", nb.p_word_given_label("boring", NEG_LABEL))
    print("boring probability given positive documents : ", nb.p_word_given_label("boring", POS_LABEL))
    print()

    print("fantastic probability given pseudocount in positive documents : ",nb.p_word_given_label_and_psuedocount("fantastic", POS_LABEL, alpha))
    print("fantastic probability given pseudocount in negative documents : ",nb.p_word_given_label_and_psuedocount("fantastic", NEG_LABEL, alpha))
    print()

    print("boring probability given pseudocount in negative documents : ",nb.p_word_given_label_and_psuedocount("boring", NEG_LABEL, alpha))
    print("boring probability given pseudocount in positive documents : ",nb.p_word_given_label_and_psuedocount("boring", POS_LABEL, alpha))
    print()

    #Question 5
    pos_accuracy,neg_accuracy,total_accuracy=nb.evaluate_classifier_accuracy(alpha)
    print("Positive class accuracy with alpha =", alpha, "is : ",pos_accuracy)
    print("Negative class accuracy with alpha = ", alpha, "is : ", neg_accuracy)
    print("Total accuracy with alpha =", alpha, "is : ", total_accuracy)
    print()
     #part 5.2

    # pseudocounts=np.arange(0.1,2.3,0.3)
    # accuracies=[]
    # for i in pseudocounts:
    #     _,_,accuracy=nb.evaluate_classifier_accuracy(i)
    #     accuracies.append(accuracy)
    #
    # plot_psuedocount_vs_accuracy(pseudocounts.tolist(),accuracies)
    print()

    #Question 6
    print("likelihood ratio of 'fantastic': ", nb.likelihood_ratio("fantastic", alpha))
    print("likelihood ratio of 'boring': ", nb.likelihood_ratio("boring", alpha))
    print()

    print("likelihood ratio of 'the': ", nb.likelihood_ratio("the", alpha))
    print("likelihood ratio of 'to': ", nb.likelihood_ratio("to", alpha))
    print()

    print ('[done.]')


def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuracies. You may want to modify this function
    to enhance your plot.
    """

    import matplotlib.pyplot as plt

    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.show()

if __name__ == '__main__':


    nb = NaiveBayes()
    nb.train_model()
    # nb.train_model(num_docs=10)
    produce_hw2_results()

