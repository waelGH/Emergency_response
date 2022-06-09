#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wael khallouli
"""
from sklearn import model_selection, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import statistics 
from sklearn.model_selection import cross_val_score


############## SVM model implementation #################################################################################################################
class SVM_Model():
  def __init__(self, kernel='linear', c=1):
    self.kernel = kernel
    self.c = c
    self.SVM = svm.SVC(C=self.c, kernel=self.kernel, degree=3, gamma='auto')
    
  def train(self,X,Y):
    self.SVM.fit(X,Y)
    
  def test(self,X):
    preds = self.SVM.predict(X)
    return preds
 
  def Calculate_Accuracy(self,preds,Y):
    # Use accuracy_score function to get the accuracy
    acc = accuracy_score(preds, Y)
    print("SVM Accuracy Score -> ",acc*100)
    return acc

  def Calculate_F_Score(self,preds,Y):
    # Use f score function to get the accuracy
    f_score = f1_score(Y, preds, labels=np.unique(preds),average='weighted')
    print(" SVM f1 Score -> ",f_score*100)
    return f_score


  def Calculate_Mathew(self,preds,Y):
    # Use mathew function to get the accuracy
    M = matthews_corrcoef(Y, preds)
    print(" SVM mathew coefficient -> ",M)
    return M

    
  def Display_classification_report(self,preds,Y):
    print(classification_report(Y, preds))

  def Calculate_confusion_Matrix(self,preds,Y):
    CM = confusion_matrix(Y, preds)
    print(CM)



############## functions to encode input text ###########################################################################################################
def Create_TFIDF(vocab,max_features = 1000,lb=1,ub=2): 
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(lb, ub), stop_words='english',max_features=max_features)
    features = tfidf.fit_transform(vocab).toarray()
    features.shape
    return tfidf

def Vector_Encoding_TFIDF(tfidf, input_X): 
    Train_X_Tfidf = tfidf.transform(input_X).toarray()
    return Train_X_Tfidf 


def Create_BOW(vocab, max_features = 1000): 
    matrix = CountVectorizer(max_features=max_features)
    BOW_features = matrix.fit_transform(vocab).toarray()
    BOW_features.shape
    return matrix

def Vector_Encoding_BOW(matrix, input_X): 
    BOW_input_X = matrix.fit_transform(input_X).toarray()
    return BOW_input_X
############### functions to encode input text ######################################################################################################




########### functions for cross-validation (k-fold) ###########################################################################################
#param_kernel = ['linear','rbf','poly']
#param_C = [0.1,0.2,0.5,1,2,3,4,5]
#param_gamma = [5,10,50]
#degree = [2,3]
          
def generate_hyper_parameter_list_SVM(param_kernel=[],param_C=[],param_gamma=[],degree=[]): 
    param_set = []
    for i in param_kernel: 
        for j in param_C: 
            for k in param_gamma: 
                for l in degree: 
                    param_set.append([i,j,k,l])
                    
    return param_set
                              
def cross_validation_svm(train_X,train_Y,param_set=[],k=5):    
    best_config = {'kernel': "" ,'C':0  ,'gamma':0  ,'degree':0}
    best_mean_acc = 0
    for e in param_set: 
        if e[0] == 'linear': 
            SVM = svm.SVC(C=e[1],kernel= e[0],gamma='auto')
        elif e[0] == 'rbf': 
            SVM = svm.SVC(C=e[1], kernel= e[0], gamma=e[2])
        elif e[0] == 'poly': 
            SVM = svm.SVC(C=e[1], kernel= e[0], degree=e[3], gamma='auto')
        
        scores = cross_val_score(SVM, train_X, train_Y,cv = k, scoring = 'f1')          
        print("configuration -->","kernel=",e[0],"-- C=",e[1],"-- gamma (rbf)=",e[2],"-- degree(poly)=",e[3]) 
        print("mean f1 --->",scores.mean())
        if(scores.mean() > best_mean_acc): 
            best_config['kernel'] = e[0]
            best_config['C'] = e[1]
            best_config['gamma'] = e[2]
            best_config['degree'] = e[3]
            best_mean_acc = scores.mean()
            
    return best_config


########### functions for cross-validation (k-fold) ###########################################################################################
   

########### functions to check stability of the SVM model ###########################################################################################
def check_variance_SVM(df,Encod_fnct,Encoding=0,nb_iter=10):          #encoding: 0->BOW    1->TFIDF
    ## change the called classifier inside #####
    ## df: must contain a column 'text' and a column 'label'
    
    ls_acc = []
    for i in range(nb_iter): 
        Train_X, Test_X, Train_Y, Test_Y = train_test_split(df['text'],df['label'],random_state=2018+i,test_size=0.2,stratify=df['label'])
    
        ###### BOW encoding ############
        if Encoding == 0:
            # create BOW encoder
            #BOW = Create_BOW(df_training['text'], 100)            
            Train_X_Encoding= Vector_Encoding_BOW(Encod_fnct,Train_X)
            Test_X_Encoding = Vector_Encoding_BOW(Encod_fnct,Test_X)
        ####################################
            
        ##### TFID encoding #############
        if Encoding == 1: 
            print('function here')
        ####################################
        
        ## fit SVM
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(Train_X_Encoding,Train_Y)
    
        predictions_SVM = SVM.predict(Test_X_Encoding)
        # Use accuracy_score function to get the accuracy
        a = accuracy_score(predictions_SVM, Test_Y)
        ls_acc.append(a)
        print("SVM Accuracy Score ",i+1,"--->",a*100,"\n")   
        
    r_list = []
    print("__________________________________________________________")
    print("average accuracy   --->",statistics.mean(ls_acc))
    r_list.append(statistics.mean(ls_acc))
    print("standard deviation -->",statistics.stdev(ls_acc))
    r_list.append(statistics.stdev(ls_acc))
    print("__________________________________________________________")

    return(r_list)
### functions to check stability of the SVM model ####################################################################################################


### Display results ################################################################################################################################
def Display_metrics(labels,predictions): 
    #preds = np.argmax(preds, axis = 1)
    acc_score = accuracy_score(labels, predictions)
    f_score = f1_score(labels, predictions, labels=np.unique(predictions),average='binary')    #average='weighted'
    M = matthews_corrcoef(labels, predictions)

    print(" model accuracy -----",acc_score) 
    print(" f1-score -----",f_score) 
    print(" Mathews coefficient -----", M) 
    
    return None


def Display_classification_report(labels,predictions): 
    print(classification_report(labels, predictions))
    
    return None


def Confusion_matrix(labels,predictions):
    return confusion_matrix(labels, predictions)

### Display results ################################################################################################################################