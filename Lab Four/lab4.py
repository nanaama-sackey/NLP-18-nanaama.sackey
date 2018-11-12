#!/usr/bin/env python
# coding: utf-8

# In[252]:


import sys
import nltk
import random
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


# In[90]:


#using pandas to handle the files.
#First read the list file and converting them to a list of dataframes using 
#Add them all into one big frame using concat
corpus=['amazon_cells_labelled.txt',"imdb_labelled.txt","yelp_labelled.txt"]
data = pd.concat(pd.read_csv(file, sep='\t', names=['Text','Label'], index_col= None, header = 0 )
for file in corpus)


# In[243]:


#Normalised form of NaiveBayes Classifier
#Training
def NaiveBayes_A():
    #noramalisation
    stop_words =set(stopwords.words("english"))
    
    #In order to count the word count per text it is easier to use Term Frequency Inverse Document Frequency
    #Tfid transforms text to feature vecors,
    vectorizer = TfidfVectorizer(use_idf=True, strip_accents ='ascii', stop_words = stop_words, lowercase=True)
    
    #Setting labels where 0 is negative and 1 is positive
    y = data.Label
    x= vectorizer.fit_transform(data.Text) #Converting the dataframes in the data into features.
    
    #splitting our data into training and test sets
    x_train, x_test, y_train, y_test =train_test_split(x,y, random_state = 35 , train_size =0.95, test_size =0.05)
    
    #Training the Classifier (MultinomialNB Normalised Classifier(MNBN Classifier))
    MNBN_classifier = MultinomialNB()
    model = MNBN_classifier.fit(x_train, y_train)
   

    return MNBN_classifier , vectorizer, model, x_test, y_test
    
#NaiveBayes_A()


# In[251]:


#TRAINING THE NORMALISED NAIVE BAIYES CLASSIFIER
def testNaiveBayes_A(file):
    #Reading of the file
    df = pd.read_csv(file, sep='\t', names=['Text','Label'], index_col=None, header =-1)
    
    #Training on the normalised nb classifier
    MNBN_classifier , vectorizer, model, x_test, y_test = NaiveBayes_A()
    
    ##Setting labels where 0 is negative and 1 is positive
    y = list(df.Label)
    x=  df.Text
    
   #Evaluating the model using the classification report function form sklearn

    predicted = model.predict(x_test)
    print("Our Accuracy is: " ,np.mean(predicted == y_test)*100) 
    
    print(" Accuracy is: " ,accuracy_score(y_test, predicted)*100) 
    
    print(" Below is how the classifier was evaluated: \n" ,classification_report(y_test, predicted))
    
    return y
testNaiveBayes_A("yelp_labelled.txt")


# In[246]:


#UnNormalised form of NaiveBayes Classifier
#Training
def NaiveBayes_B():

    #In order to count the word count per text it is easier to use Term Frequency Inverse Document Frequency
    #Tfid transforms text to feature vecors,
    vectorizer = TfidfVectorizer(use_idf=False, strip_accents =None, lowercase=False)
    
    #Setting labels where 0 is negative and 1 is positive
    y = data.Label
    x= vectorizer.fit_transform(data.Text) #Converting the dataframes in the data into features.
    
    #splitting our data into training and test sets
    x_train, x_test, y_train, y_test =train_test_split(x,y, random_state = 35 , train_size =0.95, test_size =0.05)
    
 #Training the Classifier (MultinomialNB Unormalised Classifier(MNBU Classifier))

    MNBU_classifier = MultinomialNB()
    model= MNBU_classifier.fit(x_train, y_train)
   

    return MNBU_classifier , vectorizer, model, x_test,y_test
    
#NaiveBayes_B()


# In[247]:


#TRAINING THE UNNORMALISED NAIVE BAIYES CLASSIFIER
def testNaiveBayes_B(file):
    #Reading of the file
    df = pd.read_csv(file, sep='\t', names=['Text','Label'], index_col=None, header =-1)
    
      #Training on the unnormalised nb classifier
    MNBU_classifier  , vectorizer, model, x_test, y_test = NaiveBayes_B()
    ##Setting labels where 0 is negative and 1 is positive
    y = list(df.Label)
    x=  df.Text
    
    #Evaluating the model using the classification report function form sklearn
    predicted = model.predict(x_test)
    print("Our Accuracy is: " ,np.mean(predicted == y_test)*100) 
    
    print(" Accuracy is: " ,accuracy_score(y_test, predicted)*100) 
    
    print(" Below is how the classifier is evaluated: \n" ,classification_report(y_test, predicted))
    return y
    
testNaiveBayes_B("yelp_labelled.txt")


# In[248]:


#A NORMALISED LOGISTIC REGRESSION TRAIN FUNCTION.
def normLR_A():
    #noramalisation
    stop_words =set(stopwords.words("english"))
    vectorizer = TfidfVectorizer(use_idf=True, strip_accents ='ascii', stop_words = stop_words, lowercase=True)
    
    #Setting labels where 0 is negative and 1 is positive
     
    y = data.Label
    x=vectorizer.fit_transform(data.Text)# transfroming the data to features 
    
    #Removing unnecessary punctuations
    df['Text'] = df.Text.str.replace('[^\w\s]', '') 
 
    #Training the data
    x_train, x_test, y_train, y_test =train_test_split(x,y, random_state = 40, train_size =0.95, test_size =0.05)
   
    #Training the Classifier (Logistic Regression Normalised Classifier(LRN))
    LRN_classifier = LogisticRegression()
    model = LRN_classifier.fit(x_train, y_train)
        
    return LRN_classifier, vectorizer, model, x_test, y_test
normLR_A()


# In[249]:


#TRAINING THE NORMALISED LOGISTIC REGRESSION CLASSIFIER
def testNormLR_A(file):
    #Reading of the file
    df = pd.read_csv(file, sep='\t', names=['Text','Label'], index_col=None, header =-1)
    
      #Training on the normalised LR classifier
    LRN_classifier , vectorizer, model, x_test, y_test = normLR_A()
    ##Setting labels where 0 is negative and 1 is positive
    y = list(df.Label)
    x=  df.Text
   
    #Evaluating the model using the classification report function form sklearn
    predicted = model.predict(x_test)
    print("Our Accuracy is: " ,np.mean(predicted == y_test)*100) 
    
    print(" Accuracy is: " , accuracy_score (y_test, predicted)*100) 
    
    print(" Below is how the classifier is evaluated: \n" ,classification_report(y_test, predicted))
    return y
    
    
testNormLR_A("yelp_labelled.txt")


# In[221]:


#AN UNNORMALISED  LOGISTIC REGRESSION TRAIN FUNCTION.
def unNormLR_B():
    
    #making an unnormalised form of our classifier y removing stop words, and not performing any form of normalisation
    UnNormalisedVectorizer = TfidfVectorizer(use_idf=False, strip_accents=None, lowercase=False )
    
    #Setting labels where 0 is negative and 1 is positive
    y = data.Label
    x= UnNormalisedVectorizer.fit_transform(data.Text)# transfroming the data to features 
    
    #Training the data
    x_train, x_test, y_train, y_test =train_test_split(x,y, random_state = 40, train_size =0.95, test_size =0.05)
    
    #Training the Classifier ( UnNormalised Logistic Regression Classifier)
    UnNormalisedLr_classifier =LogisticRegression()
    model = UnNormalisedLr_classifier.fit(x_train, y_train)
    
    
    return UnNormalisedLr_classifier, UnNormalisedVectorizer, model, x_test, y_test
    
  


# In[231]:


#TRAINING THE NORMALISED LOGISTIC REGRESSION CLASSIFIER
def testUnNormLR_B(file):
    #Reading of the file
    df = pd.read_csv(file, sep='\t', names=['Text','Label'], index_col=None, header =-1)
    print(df)
    
      #Training on the normalised LR classifier
        
    UnNormalisedLr_classifier, vectorizer, model, x_test, y_test = unNormLR_B()
    ##Setting labels where 0 is negative and 1 is positive
    y = list(df.Label)
    x=  df.Text
    
    #Evaluating the model using the classification report function form sklearn
    predicted = model.predict(x_test)
    print("Our Accuracy is: " ,np.mean(predicted == y_test)*100) 
    
    print(" Accuracy is: " , accuracy_score (y_test, predicted)*100) 
    
    print(" Below is how the classifier is evaluated: \n" ,classification_report(y_test, predicted))
    return y
    
    


# In[237]:


def results(classifier, version, y):
    file_name = open("results-"+classifier+"-"+version+".txt", w)
    file_name.write("y: "+"\n")
    
    for label in y:
        file_name.write(str(label)+"\n")
        
        


# In[ ]:


if__name__ == "__main__":
    classifier =sys.argv[1]
    version = sys.argv[2]
    file = sys.argv[3]
    
    if classifier == "nb" and version == "u":
        testNaiveBayes_B(file)
    elif classifier == "nb" and version == "n":
        testNaiveBayes_A(file)
    elif classifier == "lr" and version == "n":
        testNormLR_A(file)
    elif classifier == "lr" and version == "u":
        testUnNormLR_B(file)
    else:
        print("Kindly check your syntax or input the right cominations")
        
        sys.exit()
        
    
    results(classifier, version)

