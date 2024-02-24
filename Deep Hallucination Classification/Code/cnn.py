import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
import tensorflow.python.util.deprecation as deprec
from keras.models import model_from_json



# Dezactivam warning-urile care nu ne influenteaza programul
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprec._PRINT_DEPRECATION_WARNINGS = False

test_dir = "data/test"
train_and_validation_dir = "data/train+validation"
test_metadata_file = "data/test.txt"
train_metadata_file = "data/train.txt"
validation_metadata_file = "data/validation.txt"

test_images = []
predicted_labels = []
train_images = []
train_labels = []
validation_images = []
validation_labels = []



# Functii pentru incarcarea datelor din fisiere/foldere
##########################################################################################
def load_test_data(file_path):
    f = open(file_path, "r")

    f.readline()
    for test_img_data in f:
        test_img_data = test_img_data.rstrip()
        test_img_array = cv2.imread(os.path.join(test_dir, test_img_data))
        # Normalizam imaginea
        norm_test_img = (test_img_array - np.min(test_img_array)) / (np.max(test_img_array) - np.min(test_img_array))

        test_images.append(norm_test_img)

    f.close()


def load_train_data(file_path):
    f = open(file_path, "r")

    f.readline()
    for train_img_data in f:
        train_img, train_img_label = train_img_data.split(",")
        train_img_array = cv2.imread(os.path.join(train_and_validation_dir, train_img))
        # Normalizam imaginea
        norm_train_img = (train_img_array - np.min(train_img_array)) / (np.max(train_img_array) - np.min(train_img_array))

        train_images.append(norm_train_img)
        train_labels.append(int(train_img_label))

    f.close()


def load_validation_data(file_path):
    f = open(file_path, "r")

    f.readline()
    for validation_img_data in f:
        validation_img, validation_img_label = validation_img_data.split(",")
        validation_img_array = cv2.imread(os.path.join(train_and_validation_dir, validation_img))
        # Normalizam imaginea
        norm_validation_img = (validation_img_array - np.min(validation_img_array)) / (np.max(validation_img_array) - np.min(validation_img_array))

        validation_images.append(norm_validation_img)
        validation_labels.append(int(validation_img_label))

    f.close()
##########################################################################################



# Incarcarea datelor din fisiere/foldere
##########################################################################################
load_test_data(test_metadata_file)
load_train_data(train_metadata_file)
load_validation_data(validation_metadata_file)

test_images = np.array(test_images)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)
##########################################################################################



# Modelul CNN
##########################################################################################
# Optiunile pentru augmentarea datelor de antrenare
augmentOptions = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[1.0, 2.0],
    zoom_range=0.3,
    horizontal_flip=True)

# Augmentam datele de antrenare
augmentOptions.fit(x=train_images)

# Definim modelul
CNN = models.Sequential()
CNN.add(layers.Conv2D(filters=32, kernel_size=3, input_shape=(16, 16, 3), padding='same', activation='relu', kernel_initializer='random_normal'))
CNN.add(layers.BatchNormalization())
CNN.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='random_normal'))
CNN.add(layers.BatchNormalization())
CNN.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='random_normal'))
CNN.add(layers.BatchNormalization())
CNN.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='random_normal'))
CNN.add(layers.MaxPooling2D(pool_size=2))
CNN.add(layers.Dropout(rate=0.3))
CNN.add(layers.BatchNormalization())
CNN.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='random_normal'))
CNN.add(layers.MaxPooling2D(pool_size=2))
CNN.add(layers.Dropout(rate=0.3))
CNN.add(layers.BatchNormalization())
CNN.add(layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer='random_normal'))
CNN.add(layers.MaxPooling2D(pool_size=2))
CNN.add(layers.Dropout(rate=0.3))
CNN.add(layers.BatchNormalization())
CNN.add(layers.Flatten())
CNN.add(layers.Dense(units=64, activation='relu'))
CNN.add(layers.Dropout(rate=0.3, seed=69420))
CNN.add(layers.BatchNormalization())
CNN.add(layers.Dense(units=7, activation='softmax'))

# Compilam modelul
CNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Antrenam modelul
CNN.fit(x=train_images, y=train_labels, epochs=20, validation_data=(validation_images, validation_labels))

# Obtinem label-urile prezise de catre model
predicted_labels.extend(CNN.predict(x=validation_images))
# Transformam label-urile prezise de catre model, in clasele cerute
predicted_labels = np.argmax(a=predicted_labels, axis=1)

# # Populam fisierul aferent predictiilor
# fout = open("submission.txt", "w")
# fout.write("id,label\n")
# fin = open("data/test.txt", "r")
# fin.readline()
# index = 0
# for test_img in fin:
#     fout.write(test_img.rstrip() + "," + str(int(predicted_labels[index])) + "\n")
#     index += 1
# fout.close()
# fin.close()

# Evaluam modelul folosind datele de validare, si obtinem procentul
# de pierdere respectiv cel al acuratetii modelului
CNN_loss, CNN_accuracy = CNN.evaluate(x=validation_images, y=validation_labels)
print(f"\nCNN loss: {CNN_loss}")
print(f"CNN accuracy: {CNN_accuracy}\n")

# Obtinem matricea de confuzie pentru predictiile generate de catre clasificator
CNN_cm = confusion_matrix(y_true=validation_labels, y_pred=predicted_labels)
print(f"CNN confusion matrix:\n{CNN_cm}")
##########################################################################################