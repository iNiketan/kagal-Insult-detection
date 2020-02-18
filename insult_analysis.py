import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

dataset = pd.read_csv("train.csv")

# data preprocessing and cleaning
# print(dataset.iloc[:3947].Comment.isnull().sum()) # = 0
ps = PorterStemmer()


def cleaning(clum):
    clum = re.sub('[^a-zA-Z]', ' ', clum)
    clum = clum.lower()
    clum = clum.split()
    filterned_words = []
    for word in clum:
        if word not in set(stopwords.words("english")) and len(word) > 2:
            filterned_words.append(ps.stem(word))  # steming to root and append

    #filterned_words = ' '.join(filterned_words) # removed this line as w2v need single words
    return filterned_words


def corpus_creation(review):
    w_collection = []  # traverse dataset and create corpus of data
    for i in review:
        cleaned_data = cleaning(i)
        w_collection.append(cleaned_data)
    return w_collection


##################################################################
# training
#test_solution = pd.read_csv("test_with_solutions.csv")
#data_train = pd.read_csv("train.csv")


def combined_data():
    train_combined = pd.read_csv("train.csv")
    test_combined = pd.read_csv("test_with_solutions.csv")
    insult_test = test_combined['Insult']
    insult = train_combined['Insult']
    train_combined.drop(columns=['Insult', 'Date'], inplace=True)
    test_combined.drop(columns=['Insult', 'Date', 'Usage'], inplace=True)
    joined = train_combined.append(test_combined)
    joined.reset_index(inplace=True)
    return joined, insult, insult_test


#coment, y, test_y = combined_data()
#coment.drop(columns=['index'], inplace=True)
#comment = corpus_creation(coment['Comment'])


def train_test(data):
    train = data[:3947]
    test = data[3947:]
    return train, test


#train, test = train_test(comment)

"""from sklearn.feature_extraction.text import CountVectorizer


def bagofwords(data):
    cv = CountVectorizer(max_features=6500)
    sparse_matrix = cv.fit_transform(data).toarray()
    return sparse_matrix


xtrain = bagofwords(train)
xtest = bagofwords(test)

# naive bayers
from sklearn.naive_bayes import GaussianNB

naive_clf = GaussianNB()
naive_clf.fit(xtrain, y)  # 70.11 accuracy

# ypred = naive_clf.predict(xtest)

# fixing rearrangement going on....
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, ypred)
print(cm)

###############################################################
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)  # 68.45 accuracy
rfc.fit(xtrain, y)

ypred_rfc = rfc.predict(xtest)
print(confusion_matrix(test_y, ypred_rfc))"""
