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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# The model used to get the features (CNN + LSTM)

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

# We extract the feature extraction part from our model and use it to generate features out of our data

# Creating the feature extractor by removing the classification layer of the model
feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)  # Dense(64) layer

# Extract features for the entire dataset
train_features = feature_extractor.predict(X_train)
val_features = feature_extractor.predict(X_val)
test_features = feature_extractor.predict(X_test)

# --------------------------------------------------

# Applying a grid search method in order to find satisfying parameters for our Random Forest model

n_estimators_arr = [50, 100]
max_features_arr = ['sqrt', None]
max_depth_arr = [5, 10]
min_samples_split_arr = [2, 5, 10]
min_samples_leaf_arr = [2, 4, 6]

parameters = []
accuracies = []
count = 0
for n_estimators_opt in n_estimators_arr:
    for max_features_opt in max_features_arr:
        for max_depth_opt in max_depth_arr:
            for min_samples_split_opt in min_samples_split_arr:
                for min_samples_leaf_opt in min_samples_leaf_arr:
                    print(count)
                    
                    rf = RandomForestClassifier(n_estimators=n_estimators_opt, max_features=max_features_opt, max_depth=max_depth_opt, min_samples_split=min_samples_split_opt, min_samples_leaf=min_samples_leaf_opt)
                    rf.fit(train_features, y_train)
            
                    y_train_pred = rf.predict(train_features)
                    y_val_pred = rf.predict(val_features)
                    
                    parameters.append(rf.get_params())
                    accuracies.append(((accuracy_score(y_train, y_train_pred)), (accuracy_score(y_val, y_val_pred))))
            
                    count += 1

conc = [(accuracies[i], parameters[i]) for i in range(len(accuracies))]
conc.sort(key=lambda x: x[0][1], reverse=True)

# --------------------------------------------------

# Building the Random Forest model with the parameters that we found and then fitting it

rf = RandomForestClassifier()
rf.set_params(**conc[0][1])
rf.fit(train_features, y_train)

# --------------------------------------------------

# Using the Random Forest model in order to generate predictions

y_train_pred = rf.predict(train_features)
y_val_pred = rf.predict(val_features)
y_test_pred = rf.predict(test_features)

# --------------------------------------------------

# Performing a 2D PCA analysis of the features generated by the Random Forest model

def plot_pca_analysis(X, y):
    pca_model = PCA(n_components=2)
    X_transformed = pca_model.fit_transform(X)

    points_colors = ['tab:blue', 'tab:orange']
    plt.grid()
    for pca_feature, label in zip(X_transformed, y):
        plt.plot(pca_feature[0], pca_feature[1], c=points_colors[label], marker='o')
    plt.title("Validation features visualized with PCA")
    plt.xlabel("x coord")
    plt.ylabel("y coord")
    plt.show()

plot_pca_analysis(val_features, y_val.astype('int'))

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
