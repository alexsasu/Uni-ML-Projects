# Imports and downloads

import cv2 as cv
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, Conv2D, MaxPooling2D, LSTM, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('stopwords')

# --------------------------------------------------

# Loading the dataset files

data_dir = "/kaggle/input/image-sentence-pair-matching"
train_images_dir = os.path.join(data_dir, "train_images")
val_images_dir = os.path.join(data_dir, "val_images")
test_images_dir = os.path.join(data_dir, "test_images")
train_csv_path = os.path.join(data_dir, "train.csv")
val_csv_path = os.path.join(data_dir, "val.csv")
test_csv_path = os.path.join(data_dir, "test.csv")

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
test_df = pd.read_csv(test_csv_path)

# --------------------------------------------------

# Function for loading the images from folders

def load_image(image_id, images_dir):
    image_path = os.path.join(images_dir, f"{image_id}.jpg")
    image = cv.imread(image_path)
    image = image / 255.0 # We normalize the image
    return image

# --------------------------------------------------

# Storing the train, validation, and test data and labels

train_images = np.asarray([load_image(train_df.iloc[i]['image_id'], train_images_dir) for i in range(len(train_df.index))]).astype('float32')
train_texts = [train_df.iloc[i]['caption'] for i in range(len(train_df.index))]
val_images = np.asarray([load_image(val_df.iloc[i]['image_id'], val_images_dir) for i in range(len(val_df.index))]).astype('float32')
val_texts = [val_df.iloc[i]['caption'] for i in range(len(val_df.index))]
test_images = np.asarray([load_image(test_df.iloc[i]['image_id'], test_images_dir) for i in range(len(test_df.index))]).astype('float32')
test_texts = [test_df.iloc[i]['caption'] for i in range(len(test_df.index))]

y_train = np.asarray([int(train_df.iloc[i]['label']) for i in range(len(train_df.index))]).astype('float32')
y_val = np.asarray([int(val_df.iloc[i]['label']) for i in range(len(val_df.index))]).astype('float32')

# --------------------------------------------------

# Manually repairing the noisy test texts

repaired_test_texts = test_texts.copy()

for i in range(len(repaired_test_texts)):
    if repaired_test_texts[i] == "A sjmall domesticated carnivorious mammnal with sof fuh,y a sthort sout, and retracwtablbe flaws. It iw widexly kept as a pet or for catchitng mic, ad many breeds zhlyde beefn develvoked.":
        repaired_test_texts[i] = "A small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws. It is widely kept as a pet or for catching mice, and many breeds have been developed."
    elif repaired_test_texts[i] == "A smafml vessef epropoeilled on watvewr by ors, sauls, or han engie.":
        repaired_test_texts[i] = "A small vessel propelled on water by oars, sails, or an engine."
    elif repaired_test_texts[i] == "An instqrumemnt used for cutting cloth, paper, axdz othr thdin mteroial, consamistng of two blades lad one on tvopb of the other and fhastned in tle mixdqdjle so as to bllow them txo be pened and closed by thumb and fitngesr inserted tgrough rings on kthe end oc thei vatndlzes.":
        repaired_test_texts[i] = "An instrument used for cutting cloth, paper, and other thin material, consisting of two blades laid one on top of the other and fastened in the middle so as to allow them to be opened and closed by a thumb and finger inserted through rings on the end of their handles."

test_texts = repaired_test_texts.copy()

# --------------------------------------------------

# Methods for preprocessing the text part of the samples

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove stop words
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Preprocessing our texts
train_texts = [preprocess_text(text) for text in train_texts]
val_texts = [preprocess_text(text) for text in val_texts]
test_texts = [preprocess_text(text) for text in test_texts]

# --------------------------------------------------

# Fitting and applying a TensorFlow tokenizer on our texts

tokenizer = Tokenizer(oov_token='<UNK>')
tokenizer.fit_on_texts(train_texts)

max_seq_length = 60 # We want each resulted sequence to be of max length 60
train_sequences = tokenizer.texts_to_sequences(train_texts)
padded_train_sequences = np.asarray(pad_sequences(train_sequences, maxlen=max_seq_length, truncating='post')).astype('float32')
val_sequences = tokenizer.texts_to_sequences(val_texts)
padded_val_sequences = np.asarray(pad_sequences(val_sequences, maxlen=max_seq_length, truncating='post')).astype('float32')
test_sequences = tokenizer.texts_to_sequences(test_texts)
padded_test_sequences = np.asarray(pad_sequences(test_sequences, maxlen=max_seq_length, truncating='post')).astype('float32')

# --------------------------------------------------

# Aggregating the train, validation, and test data in order to feed it to the model

X_train = [train_images, padded_train_sequences]
X_val = [val_images, padded_val_sequences]
X_test = [test_images, padded_test_sequences]

# --------------------------------------------------

# The model used (CNN + LSTM)

# Model parameters
image_shape = (100, 100, 3)
vocab_size = len(tokenizer.word_index) + 1   # Size of the vocabulary
embedding_dim = 50                           # Embedding dimension for text
max_seq_length = 60                          # Maximum text sequence length

# Image subnetwork
def build_image_subnet(input_shape):
    image_input = Input(shape=input_shape, name="Image_Input")
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    image_features = Dense(64, activation='relu')(x)
    image_features = Dropout(0.8)(image_features)
    return Model(image_input, image_features, name="Image_Subnet")

# Text subnetwork
def build_text_subnet(vocab_size, embedding_dim, max_seq_length):
    text_input = Input(shape=(max_seq_length,), name="Text_Input")
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length)(text_input)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.25)(x)
    text_features = Dense(64, activation='relu')(x)
    text_features = Dropout(0.8)(text_features)
    return Model(text_input, text_features, name="Text_Subnet")

# Combined model
def build_combined_model(image_shape, vocab_size, embedding_dim, max_seq_length):
    image_subnet = build_image_subnet(image_shape)
    text_subnet = build_text_subnet(vocab_size, embedding_dim, max_seq_length)
    
    # Combined features
    combined_features = Concatenate()([image_subnet.output, text_subnet.output])
    x = Dense(64, activation='relu')(combined_features)
    x = Dropout(0.8)(x)
    output = Dense(1, activation='sigmoid', name="Classifier")(x)
    
    # Final model
    model = Model(inputs=[image_subnet.input, text_subnet.input], outputs=output, name="Image_Text_Classifier")
    return model

# --------------------------------------------------

# We build and compile our model

model = build_combined_model(image_shape, vocab_size, embedding_dim, max_seq_length)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# --------------------------------------------------

# We train our model

early_stopping = EarlyStopping(patience=3, restore_best_weights=True) # We apply early stopping to mitigate overfitting
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), callbacks=[early_stopping])

# --------------------------------------------------

# After training the model, we want to plot its accuracy and loss during the fitting process

def generate_plots(history):
    # Plotting the train and validation accuracy history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.show()
    
    # Plotting the train and validation loss history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.show()

generate_plots(history)

# --------------------------------------------------

# We obtain our train, validation, and test predictions

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# We convert the model predictions into the correct 0/1 format
y_train_pred = np.asarray([round(prediction[0]) for prediction in y_train_pred]).astype('float32')
y_val_pred = np.asarray([round(prediction[0]) for prediction in y_val_pred]).astype('float32')
y_test_pred = np.asarray([round(prediction[0]) for prediction in y_test_pred]).astype('float32')

# --------------------------------------------------

# We display different metrics based on our predicted data, and we also show confusion matrices

def display_results(y_true, y_pred):
    print(classification_report(y_true, y_pred, labels=[0, 1]))
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)
    sns.heatmap(conf_mat, annot=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

display_results(y_train, y_train_pred)
display_results(y_val, y_val_pred)

# --------------------------------------------------

# We generate the submission file based on our predictions

test_ids = [test_df.iloc[i]['id'] for i in range(len(test_df.index))]
with open("submission_file.csv", "w") as submission_file:
    submission_file.write("id,label" + '\n')
    for i in range(len(X_test[0])):
        submission_file.write(f"{test_ids[i]},{str(int(y_test_pred[i]))}\n")
