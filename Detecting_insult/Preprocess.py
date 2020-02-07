import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

data = pd.read_csv("train.csv")

# data preprocessing and cleaning
print(data.iloc[:3947].Comment.isnull().sum())

ps = PorterStemmer()


class Preprocess:
    def __init__(self):
        pass

    def cleaning(clum):
        clum = re.sub('[^a-zA-Z]', ' ', clum)
        clum = clum.lower()
        clum = clum.split()
        filterned_words = []
        for word in clum:
            if word not in set(stopwords.words("english")) and len(word) > 2:
                filterned_words.append(ps.stem(word))  # steming to root and append

        filterned_words = ' '.join(filterned_words)
        return filterned_words

    def corpus_creation(review):
        w_collection = []
        for i in review:
            cleaned_data = Preprocess.cleaning(i)
            w_collection.append(cleaned_data)
        return w_collection

