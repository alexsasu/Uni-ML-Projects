# Imports

import matplotlib.pyplot as plt

import json
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

import re
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

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
label_types = [0, 1, 2, 3]

train_data = [sample["sentence1"] + sentences_separator + sample["sentence2"] for sample in train_samples]
train_labels = [sample["label"] for sample in train_samples]

val_data = [sample["sentence1"] + sentences_separator + sample["sentence2"] for sample in val_samples]
val_labels = [sample["label"] for sample in val_samples]

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

# Obtaining the features resulted from applying the TF-IDF method on data

features_count = 5000

tfidf_vectorizer = TfidfVectorizer(
    preprocessor=preprocess_data,
    max_features=features_count
)

tfidf_vectorizer.fit(train_data)

train_features = tfidf_vectorizer.transform(train_data).toarray()
val_features = tfidf_vectorizer.transform(val_data).toarray()
test_features = tfidf_vectorizer.transform(test_data).toarray()

# --------------------------------------------------

# Creating the model

rf = RandomForestClassifier(class_weight='balanced')

# --------------------------------------------------

# Training the model

rf.fit(train_features, train_labels)

# --------------------------------------------------

# Using the model to generate predictions

val_labels_pred = rf.predict(val_features)
test_labels_pred = rf.predict(test_features)

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
