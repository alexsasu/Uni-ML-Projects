# Imports

import copy

import json

import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import string

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

train_data = [sample["sentence1"] + sentences_separator + sample["sentence2"] for sample in train_samples]
train_labels = [sample["label"] for sample in train_samples]

val_data = [sample["sentence1"] + sentences_separator + sample["sentence2"] for sample in val_samples]
val_labels = [sample["label"] for sample in val_samples]

train_and_val_data = copy.copy(train_data)
train_and_val_data.extend(val_data)
train_and_val_labels = copy.copy(train_labels)
train_and_val_labels.extend(val_labels)

test_data = [sample["sentence1"] + sentences_separator + sample["sentence2"] for sample in test_samples]
test_guids = [sample["guid"] for sample in test_samples]

# --------------------------------------------------

# Implementing the text preprocessing methods

def preprocess_data(data):
    data = data.lower()
    data = re.sub(f"[{re.escape(string.punctuation)}]", "", data)

    return data

# --------------------------------------------------

# Preprocessing the data

train_and_val_data = [preprocess_data(data) for data in train_and_val_data]
test_data = [preprocess_data(data) for data in test_data]

# --------------------------------------------------

# Obtaining the features resulted from applying the TF-IDF method on data

features_count = 5000

tfidf_vectorizer = TfidfVectorizer(
    preprocessor=preprocess_data,
    max_features=features_count
)

tfidf_vectorizer.fit(train_and_val_data)

train_and_val_features = tfidf_vectorizer.transform(train_and_val_data).toarray()
test_features = tfidf_vectorizer.transform(test_data).toarray()

# --------------------------------------------------

# Creating the model

rf = RandomForestClassifier(class_weight='balanced')

# --------------------------------------------------

# Training the model

rf.fit(train_and_val_features, train_and_val_labels)

# --------------------------------------------------

# Using the model to generate predictions

test_labels_pred = rf.predict(test_features)

# --------------------------------------------------

# Populating the submission file

header = "guid,label"

with open("submission.csv", "w") as output_file:
    output_file.write(header + '\n')
    for i in range(len(test_guids)):
        output_file.write(test_guids[i] + ',' + str(test_labels_pred[i]) + '\n')
