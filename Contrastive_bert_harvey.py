########################################################
############### Imports ################################
########################################################
from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

import pickle
import numpy as np
from numpy import mean
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from keras.layers import LSTM
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import collections
from collections import Counter

from sklearn.model_selection import train_test_split
import string

import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))

### imports (1) ##
import pandas as pd
import numpy as pn
from numpy import mean

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import string

import collections
from collections import Counter

### imports (2) ##
from string import punctuation
from os import listdir
from numpy import array

from pickle import load
from numpy import array

### imports (4) ##
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

import matplotlib.pyplot as plt

import tensorflow_hub as hub
import tensorflow_text as text


##################################################
#### functions to clean the corpus ###############
##################################################
def remove_punc(text): 
    ### only for logical feature
    string.punctuation = '!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~'
    text = "".join([char for char in text if char not in string.punctuation ])
    #text = re.sub('[0-9]+', '', text)
    return text

def tokenization(text):
    #text = re.split('\W+',text)
    text = re.split('\s+',text)
    return text

def remove_url(text):
    text = re.sub(r'http\S+', '', text)
    return(text)

def lower_case(text):
    text = text.lower()
    return(text)

def remove_stopwords(text):
    text = [word for word in text if word not in STOPWORDS]
    return text


def clean_text(text): 
    # lower case
    text_lower = lower_case(text)
    
    # remove puntuation
    text_punc = remove_punc(text_lower) 
    
    #remove URLS
    text_url = remove_url(text_punc)
    
    # tokenization
    text_tokens = tokenization(text_url)
    
    # remove stop words
    no_stop_tokens = remove_stopwords(text_tokens)
    
    return no_stop_tokens


########################################################################################################
#########  function to clean and tokenize a corpus (to create a vocab given a corpus)  #################
###        input: corpus         output: clean tokens ##################################################
########################################################################################################
def clean_corpus(corpus):
    # convert all to lower case 
    corpus_lower = lower_case(corpus)    
    # remove punctuation
    corpus_punc = remove_punc(corpus_lower) 
    # remove punctuation from each token
    #table = str.maketrans('', '', string.punctuation)
    #tokens = [w.translate(table) for w in tokens]
    
    #remove URLS
    corpus_url = remove_url(corpus_punc)
    
    # tokenization
    corpus_tokens = tokenization(corpus_url)
    # split into tokens by white space
    #tokens = corpus.split()
    
    # remove stop words
    no_stop_tokens = remove_stopwords(corpus_tokens)
    # filter out stop words
    # stop_words = set(stopwords.words('english'))
    # tokens = [w for w in tokens if not w in stop_words]
    
    # remove remaining tokens that are not alphabetic
    #tokens = [word for word in tokens if word.isalpha()]
    
    # filter out short tokens
    final_tokens = [word for word in no_stop_tokens if len(word) > 1]

    return final_tokens

########################################################################################################
#########  function to clean and tokenize a document based on a given vocabulary   #####################
###        input: document         output: clean document's tokens #####################################
########################################################################################################
def clean_document_vocab(doc, vocab):
    # convert to lower case 
    doc_lower = lower_case(doc)
    
    # remove punctuation
    doc_punc = remove_punc(doc_lower) 

    #remove URLS
    doc_url = remove_url(doc_punc)
    
    # tokenization
    doc_tokens = tokenization(doc_url)
    
    # remove stop words
    doc_no_stop_tokens = remove_stopwords(doc_tokens)
    
    # filter out tokens not in vocab ---> add or w=='<LOC'>
    tokens = [w for w in doc_no_stop_tokens if w in vocab]
        
    #tokens = ' '.join(tokens)
    return tokens

########################################################################################################
#########  Creat BERT-CNN encoder   ####################################################################
########################################################################################################
def create_kim_BERT_encoder(input_shape=(128,768,)):
    
    inputs1 = Input(shape=input_shape)  

    ### channel 1 #####
    conv1 = Conv1D(filters=1024, kernel_size=2, activation='relu')(inputs1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=8)(drop1)
    flat1 = Flatten()(pool1)

    ### channel 2 #####
    conv2 = Conv1D(filters=128, kernel_size=16, activation='relu')(inputs1)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=6)(drop2)
    flat2 = Flatten()(pool2)
    
    
    ### channel 3 #####
    conv3 = Conv1D(filters=16, kernel_size=14, activation='relu')(inputs1)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=6)(drop3)
    flat3 = Flatten()(pool3)

    # merge
    union = concatenate([flat1, flat2, flat3])
    
    ###  add layer: 2048-d
    #final_embed = layers.Dense(2048, activation="relu")(union) 
    model = tf.keras.Model(inputs=inputs1, outputs=union)
  
    return model

########################################################################################################
#########  Projection and contrastive loss function   ##################################################
########################################################################################################
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def add_projection_head(encoder,input_shape=(128, 768),):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs1= layers.Dense(128, activation="relu")(features)     
    outputs = layers.Dense(32, activation="relu")(outputs1)
    
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="text-classification-encoder_with_projection-head-32"
    )
    return model


###############################################################################
###### Classifier on top of the encoder #######################################
###############################################################################
def create_classifier_bert_contrastive(encoder,params = {},input_shape = (128,768,),num_classes = 2,trainable=False):

    dropout_rate = params['dropout_rate']
    hidden_units = params['hidden_units'] 
    
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="text_classifier_bert_encoder")
    
    return model


###############################################################################
###### main function ##########################################################
###############################################################################
def main():
    ## Path data
    file_path = '/home/wkhal001/Desktop/data_rescue_mining/labeled_ds_Corrected_csv.csv'

    #### Read data set 
    initial_labeledDF = pd.read_csv(file_path) 
    initial_labeledDF = initial_labeledDF.drop(['id','Unnamed: 0','loc','situ','save','sos','address','sos.pred'],1)
    Training_set = initial_labeledDF[['text','sos.correct']]
    print('Data Set read.....')
    
    #### Create vocabulary
    corpus = ''
    for i in range(len(Training_set)): #Training_set_masked
        st = Training_set.loc[i, "text"]
        corpus = corpus + " " + st
    
    #### corpus
    vocabulary = clean_corpus(corpus)

    #### count vocabulary with collection counter ####
    vocab = Counter()
    vocab.update(vocabulary)
    print('Vocabulary created...',len(vocab),' tokens')

    ## load Bert Models
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    
    iteration = 1
    Results = {'f1':[],'recall':[],'precision':[]}
    for i in range(10): 
        
        print('-------------------------------------------------------')
        print('------------------- Iteration ',iteration,'------------')
        print('-------------------------------------------------------')
        train_text, test_text, train_labels, test_labels = train_test_split(Training_set, Training_set['sos.correct'], 
                                                                    random_state=2018+i,   
                                                                    test_size=0.2, 
                                                                    stratify=Training_set['sos.correct'])
        
        ######### Load and clean train and test sets #########
        print('load/clean train and test docs...')
        train_documents = list()
        for index,row in train_text.iterrows(): 
            tweet = row["text"]
            tokens = clean_document_vocab(tweet, vocab)
            joined_tokens = ' '.join(tokens)
            train_documents.append(joined_tokens)

        test_documents = list()
        for index,row in test_text.iterrows(): 
            tweet = row["text"]
            tokens = clean_document_vocab(tweet, vocab)
            joined_tokens = ' '.join(tokens)
            test_documents.append(joined_tokens)
            
        ######### convert train and test sets to Bert embeddings ########
        print('BERT -- encode training data...')
        preprocessed_text = bert_preprocess(train_documents)
        encoded_train = bert_encoder(preprocessed_text)['encoder_outputs'][0] 
        print('All training tweets encoded')
        
        print('BERT -- encode testing data...')
        preprocessed_text = bert_preprocess(test_documents)
        encoded_test = bert_encoder(preprocessed_text)['encoder_outputs'][0] 
        print('All testing tweets encoded')
        
        
        #### Build encoder model ##############################
        encoder = create_kim_BERT_encoder((128,768,))
        
        #### Build projection head ############################
        encoder_with_projection_head = add_projection_head(encoder)
        encoder_with_projection_head.summary()
        
        #### Train contrastive #################################
        params_contrastive = {'lr':0.0005,'batch_size':512,'num_epochs':500,'dropout_rate':0.5,'temp':0.005} 
        
        encoder_with_projection_head.compile(
            optimizer=keras.optimizers.Adam(params_contrastive['lr']), ###param 1
            loss=SupervisedContrastiveLoss(params_contrastive['temp']), ###param 2
        )
        print('Encoder + projection created and compiled...')
        
        ##### Convert data to array ############################
        training_padded = np.array(encoded_train)
        training_labels = np.array(train_labels)

        testing_padded = np.array(encoded_test)
        testing_labels = np.array(test_labels)
        
        
        #### train the contrastive learning framework #####
        print('Train contrastive ...')
        history = encoder_with_projection_head.fit(
            x=training_padded, y=training_labels, batch_size=params_contrastive['batch_size'], epochs=params_contrastive['num_epochs']
        )

        #### Create classifier on top of the encoder #########
        params_classifier = {'lr':0.01,'batch_size':128,'num_epochs':150,'dropout_rate':0.2,'nb_class':2,'hidden_units':64} 
        input_shape = (128,768,)
        classifier = create_classifier_bert_contrastive(encoder,params_classifier,input_shape,params_classifier['nb_class'])
        classifier.summary()
        
        classifier.compile(
            optimizer=keras.optimizers.Adam(params_classifier['lr']),
            loss=keras.losses.SparseCategoricalCrossentropy(), 
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], 
        )

        print('Train classifier.......')
        history = classifier.fit(x=training_padded, y=training_labels, batch_size=params_classifier['batch_size'], epochs=params_classifier['num_epochs']) 
        
        ##### Predict and evaluate #####################
        predictions = classifier.predict(testing_padded)
        preds = np.argmax(predictions, axis=1)

        f1 = f1_score(testing_labels, preds, average='binary')
        precision = precision_score(testing_labels, preds, average='binary')
        recall = recall_score(testing_labels, preds, average='binary')

        Results['f1'].append(f1)
        Results['recall'].append(recall)
        Results['precision'].append(precision)

        print('Testing -- class 1 f1 score ..',f1)
        print('Testing -- class 1 precision ..',precision)
        print('Testing -- class 1 recall ..',recall)
        print('\n')    
        iteration = iteration + 1
    
    ##### Save results after all iterations ########
    with open('/home/wkhal001/Crisis-classification/Domain_Adaptation/Contrastive-Keras-4/BERT-CNN-Experiments/BERT_cnn1.json', 'wb') as fp:
        pickle.dump(Results, fp)
    
    with open('/home/wkhal001/Crisis-classification/Domain_Adaptation/Contrastive-Keras-4/BERT-CNN-Experiments/BERT_cnn1_params_contrastive.json', 'wb') as fp1:
        pickle.dump(params_contrastive, fp1)
    
    with open('/home/wkhal001/Crisis-classification/Domain_Adaptation/Contrastive-Keras-4/BERT-CNN-Experiments/BERT_cnn1_params_classifier.json', 'wb') as fp2:
        pickle.dump(params_classifier, fp2)

if __name__ == "__main__":
    main()
