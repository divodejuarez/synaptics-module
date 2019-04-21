# use natural language toolkit
import nltk
# nltk.download('punkt') # run this line if is the first time if this script is run

import os
import json
import datetime
import pandas as pd
import numpy as np
import time

from nltk.stem.lancaster import LancasterStemmer


class Neural_Network():
    def __init__(self, csv_name):
        self.stemmer    = LancasterStemmer()
        self.df         = pd.read_csv(csv_name)
        self.pre_process_data()
        self.stem()
        self.train_set()
        self.X = np.array(self.training)
        self.y = np.array(self.output)

        start_time = time.time()

        # train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)
        self.train(self.X, self.y, hidden_neurons=20, alpha=0.01, epochs=1000, dropout=False, dropout_percent=0.2)

        elapsed_time = time.time() - start_time
        print ("processing time:", elapsed_time, "seconds")

        # probability threshold
        self.ERROR_THRESHOLD = 0.2
        # load our calculated synapse values
        self.synapse_file = 'synapses.json' 
        with open(self.synapse_file) as data_file: 
            self.synapse = json.load(data_file) 
            self.synapse_0 = np.asarray(self.synapse['synapse0']) 
            self.synapse_1 = np.asarray(self.synapse['synapse1'])

    def pre_process_data(self):
        self.training_data = []
        for row in self.df.itertuples():
            if row.category_id == 0:
                self.training_data.append({"class":row.category, "sentence":row.commands})
            elif row.category_id == 1:
                self.training_data.append({"class":row.category, "sentence":row.commands})
            else:
                self.training_data.append({"class":row.category, "sentence":row.commands})
    
    def stem(self):
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!']
        for pattern in self.training_data:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern['sentence'])
            # add to our words list
            self.words.extend(w)
            # add to documents in our corpus
            self.documents.append((w, pattern['class']))
            # add to our classes list
            if pattern['class'] not in self.classes:
                self.classes.append(pattern['class'])

        # stem and lower each word and remove duplicates
        self.words = [self.stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = list(set(self.words))

        # remove duplicates
        self.classes = list(set(self.classes))
    
    def train_set(self):
        # create our training data
        self.training = []
        self.output = []
        # create an empty array for our output
        self.output_empty = [0] * len(self.classes)

        # training set, bag of words for each sentence
        for doc in self.documents:
            # initialize our bag of words
            self.bag = []
            # list of tokenized words for the pattern
            self.pattern_words = doc[0]
            # stem each word
            self.pattern_words = [self.stemmer.stem(word.lower()) for word in self.pattern_words]
            # create our bag of words array
            for w in self.words:
                self.bag.append(1) if w in self.pattern_words else self.bag.append(0)

            self.training.append(self.bag)
            # output is a '0' for each tag and '1' for current tag
            self.output_row = list(self.output_empty)
            self.output_row[self.classes.index(doc[1])] = 1
            self.output.append(self.output_row)
    
    
    # Helpful functions
    # compute sigmoid nonlinearity
    def sigmoid(self, x):
        output = 1/(1+np.exp(-x))
        return output

    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)
    
    def clean_up_sentence(self, sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return(np.array(bag))

    def think(self, sentence, show_details=False):
        x = self.bow(sentence.lower(), self.words, show_details)
        if show_details:
            print ("sentence:", sentence, "\n bow:", x)
        # input layer is our bag of words
        l0 = x
        # matrix multiplication of input and hidden layer
        l1 = self.sigmoid(np.dot(l0, self.synapse_0))
        # output layer
        l2 = self.sigmoid(np.dot(l1, self.synapse_1))
        return l2

    # Train Neural Model
    def train(self, X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

        print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
        print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(self.X),len(self.X[0]),1, len(self.classes)) )
        np.random.seed(1)

        last_mean_error = 1
        # randomly initialize our weights with mean 0
        synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
        synapse_1 = 2*np.random.random((hidden_neurons, len(self.classes))) - 1

        prev_synapse_0_weight_update = np.zeros_like(synapse_0)
        prev_synapse_1_weight_update = np.zeros_like(synapse_1)

        synapse_0_direction_count = np.zeros_like(synapse_0)
        synapse_1_direction_count = np.zeros_like(synapse_1)
            
        for j in iter(range(epochs+1)):

            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0, synapse_0))
                    
            if(dropout):
                layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

            layer_2 = self.sigmoid(np.dot(layer_1, synapse_1))

            # how much did we miss the target value?
            layer_2_error = y - layer_2

            if (j% 10000) == 0 and j > 5000:
                # if this 10k iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                    last_mean_error = np.mean(np.abs(layer_2_error))
                else:
                    print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                    break
                    
            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * self.sigmoid_output_to_derivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * self.sigmoid_output_to_derivative(layer_1)
            
            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
            
            if(j > 0):
                synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
            
            synapse_1 += alpha * synapse_1_weight_update
            synapse_0 += alpha * synapse_0_weight_update
            
            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update

        now = datetime.datetime.now()

        # persist synapses
        synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
                'datetime': now.strftime("%Y-%m-%d %H:%M"),
                'words': self.words,
                'classes': self.classes
                }
        synapse_file = "synapses.json"

        with open(synapse_file, 'w') as outfile:
            json.dump(synapse, outfile, indent=4, sort_keys=True)
        print ("saved synapses to:", synapse_file)

    def classify(self, sentence, show_details=False):
        results = self.think(sentence, show_details)

        results = [[i,r] for i,r in enumerate(results) if r>self.ERROR_THRESHOLD ] 
        results.sort(key=lambda x: x[1], reverse=True) 
        return_results =[[self.classes[r[0]],r[1]] for r in results]
        print ("%s \n classification: %s" % (sentence, return_results))
        return return_results

neural = Neural_Network('ConjuntoDatos-Completo.csv')
neural.classify("esta activado el control parental?")
neural.classify("mete esa pelicula a mis favoritos")
neural.classify("cambiale a la temporada dos")
neural.classify("activa el menu de la programacion")
neural.classify("cuando tengo que pagar netflix?")
neural.classify("activa la consola de videojuegos")
# print()
# classify("colocame un recordatorio para Lucifer a las nueve de la noche", show_details=True)