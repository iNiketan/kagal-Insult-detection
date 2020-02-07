from Preprocess import Preprocess as Ipreprocess
import pandas as pd
import numpy as np

test_solution = pd.read_csv("test_with_solutions.csv")
data_train = pd.read_csv("train.csv")


def combined_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test_with_solutions.csv")
    insult_test = test['Insult']
    insult = train['Insult']
    train.drop(columns=['Insult', 'Date'], inplace=True)
    test.drop(columns=['Insult', 'Date', 'Usage'], inplace=True)
    joined = train.append(test)
    joined.reset_index(inplace=True)
    return joined, insult, insult_test


coment, y, test_y = combined_data()
coment.drop(columns=['index'], inplace=True)
comment = Ipreprocess.corpus_creation(coment['Comment'])


def train_test(data):
    train = data[:3947]
    test = data[3947:]
    return train, test


train, test = train_test(comment)

from sklearn.feature_extraction.text import CountVectorizer
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

ypred = naive_clf.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, ypred)
print(cm)

###############################################################
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)  # 68.45 accuracy
rfc.fit(xtrain, y)

ypred_rfc = rfc.predict(xtest)
print(confusion_matrix(test_y, ypred_rfc))

