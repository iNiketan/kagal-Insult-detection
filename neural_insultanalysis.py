import keras.backend as K
import multiprocessing
import tensorflow as tf
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from insult_analysis import corpus_creation, combined_data

# Set random seed (for reproducibility)
np.random.seed(1000)

use_gpu = True

config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(),
                        inter_op_parallelism_threads=multiprocessing.cpu_count(),
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 1 if use_gpu else 0})

session = tf.Session(config=config)
K.set_session(session)

# data import and cleaning
test_solution = pd.read_csv("test_with_solutions.csv")
data_train = pd.read_csv("train.csv")


corpus = corpus_creation(data_train['Comment'])


# gensim word2vec model
vector_size = 512
window_size = 10

# Create Word2Vec
word2vec = Word2Vec(sentences=corpus,
                    size=vector_size,
                    window=window_size,
                    negative=20,
                    iter=50,
                    seed=1000,
                    workers=multiprocessing.cpu_count())

# Train subset size (0 < size < len(tokenized_corpus))
train_size = 3900
# test len(corpus - train)
test_size = 47

# Compute average and max tweet length
avg_length = 0.0
max_length = 0

for comment in corpus:
    if len(comment) > max_length:
        max_length = len(comment)
    avg_length += float(len(comment))

print('Average tweet length: {}'.format(avg_length / float(len(corpus))))
print('Max tweet length: {}'.format(max_length))

max_comment_length = 1359

# create train and test set
# generate random indexes











