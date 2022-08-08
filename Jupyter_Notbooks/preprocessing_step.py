import string
import re
import nltk 

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')

from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))

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


# def clean_text(text): 
#     # lower case
#     text_lower = lower_case(text)
    
#     # remove puntuation
#     text_punc = remove_punc(text_lower) 
    
#     #remove URLS
#     text_url = remove_url(text_punc)
    
#     # tokenization
#     text_tokens = tokenization(text_url)
    
#     # remove stop words
#     no_stop_tokens = remove_stopwords(text_tokens)
    
#     return no_stop_tokens


###### CLEAN TEXT FUNCTION #############
##   input: text
##   output: cleaned tokens
#######################################
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