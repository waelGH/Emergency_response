import string
import re
import nltk 

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')

def remove_punc(text): 
    text = "".join([char for char in text if char not in string.punctuation ])
    text = re.sub('[0-9]+', '', text)
    return text


def tokenization(text):
    text = re.split('\W+',text)
    return text


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text


###### CLEAN TEXT FUNCTION #############
##   input: text
##   output: cleaned tokens
#######################################
def clean_text(text): 
    # remove puntuation
    text_punc = remove_punc(text) 
    # tokenization
    text_tokens = tokenization(text_punc)
    # remove stop words
    no_stop_tokens = remove_stopwords(text_tokens)
    
    return no_stop_tokens


def Display_evaluations(labels,predictions): 
    print('here')


