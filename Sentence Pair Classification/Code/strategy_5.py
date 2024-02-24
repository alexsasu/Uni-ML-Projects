# Imports

from gensim.models import Word2Vec as w2v

import json

import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

import numpy as np

import re
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import string

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Dropout, Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------

# Loading the jsons that contain the train, validation, and test data

train_file = open('train.json')
val_file = open('val.json')
test_file = open('test.json')

train_samples = json.load(train_file)
val_samples = json.load(val_file)
test_samples = json.load(test_file)

train_file.close()
val_file.close()
test_file.close()

# --------------------------------------------------

# Extracting the data from the jsons that we will need for training and testing
# the model, as well as for generating the test labels

sentences_separator = "  SENTENCESEPARATOR  "
label_types = [0, 1, 2, 3]

train_data = [sample["sentence1"] + sentences_separator + sample["sentence2"] for sample in train_samples]
train_labels = [sample["label"] for sample in train_samples]
numpy_train_labels = np.asarray(train_labels)

val_data = [sample["sentence1"] + sentences_separator + sample["sentence2"] for sample in val_samples]
val_labels = [sample["label"] for sample in val_samples]
numpy_val_labels = np.asarray(val_labels)

test_data = [sample["sentence1"] + sentences_separator + sample["sentence2"] for sample in test_samples]
test_guids = [sample["guid"] for sample in test_samples]

# --------------------------------------------------

# Implementing the text preprocessing methods

stop_words = set(stopwords.words('romanian'))
def remove_stopwords(data):
    data = data.split()

    preprocessed_data = []

    for word in data:
        if word not in stop_words:
            preprocessed_data.append(word)

    preprocessed_data = " ".join(preprocessed_data)

    return preprocessed_data

def preprocess_data(data):
    data = data.lower()
    data = re.sub(f"[{re.escape(string.punctuation)}]", "", data)
    data = remove_stopwords(data)

    return data

# --------------------------------------------------

# Preprocessing the data

train_data = [preprocess_data(data) for data in train_data]
val_data = [preprocess_data(data) for data in val_data]
test_data = [preprocess_data(data) for data in test_data]

# --------------------------------------------------

# Applying the Keras Tokenizer on data to obtain the features for the model

tokenizer = Tokenizer(oov_token="?UNKNOWN?")
tokenizer.fit_on_texts(train_data)
vocab_len = len(tokenizer.word_index) + 1

train_sequences = tokenizer.texts_to_sequences(train_data)
val_sequences = tokenizer.texts_to_sequences(val_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

# The sequences need to have the same length before entering the model
max_seq_len = 200
padded_train_sequences = pad_sequences(train_sequences, maxlen=max_seq_len, padding='post', truncating="post")
padded_val_sequences = pad_sequences(val_sequences, maxlen=max_seq_len, padding='post', truncating="post")
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_seq_len, padding='post', truncating="post")

# --------------------------------------------------

# Using Word2Vec to generate embeddings for the model

embedding_len = 100

train_data_split = [data.split() for data in train_data]

w = w2v(
    train_data_split,
    vector_size=embedding_len,
    sg=1
)

embeddings = np.zeros((vocab_len, embedding_len))
for word, i in tokenizer.word_index.items():
    if word in w.wv:
        embeddings[i] = w.wv[word]

# --------------------------------------------------

# Building the model

lstm = Sequential()
lstm.add(Embedding(input_dim=vocab_len, output_dim=embedding_len, weights=[embeddings], input_length=max_seq_len, trainable=False))
lstm.add(Bidirectional(LSTM(units=128, return_sequences=True)))
lstm.add(Dropout(0.8))
lstm.add(Bidirectional(LSTM(units=128)))
lstm.add(Dropout(0.8))
lstm.add(Dense(4, activation='softmax'))

# --------------------------------------------------

# Compiling the model

lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --------------------------------------------------

# Training the model

custom_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.asarray(label_types), y=train_labels)
custom_weights = {i : custom_weights[i] for i, label in enumerate(np.asarray(label_types))}

callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = lstm.fit(padded_train_sequences, numpy_train_labels, epochs=15, batch_size=1024, class_weight=custom_weights, callbacks=[callback], validation_data=(padded_val_sequences, numpy_val_labels))

# --------------------------------------------------

# Using the model to generate predictions

val_labels_pred = [np.argmax(pred) for pred in lstm.predict(padded_val_sequences)]
test_labels_pred = [np.argmax(pred) for pred in lstm.predict(padded_test_sequences)]

# --------------------------------------------------

# Showing results

print(classification_report(val_labels, val_labels_pred, labels=label_types))

print()

cm = confusion_matrix(val_labels, val_labels_pred)
print(cm)
sns.heatmap(cm, annot=True)
plt.ylabel('True', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

# --------------------------------------------------

# Populating the submission file

header = "guid,label"

with open("submission.csv", "w") as output_file:
    output_file.write(header + '\n')
    for i in range(len(test_guids)):
        output_file.write(test_guids[i] + ',' + str(test_labels_pred[i]) + '\n')
