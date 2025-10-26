import os
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm



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
        test_img_array = cv2.imread(os.path.join(test_dir, test_img_data), cv2.IMREAD_GRAYSCALE)
        # Normalizam imaginea
        norm_test_img = (test_img_array - np.min(test_img_array)) / (np.max(test_img_array) - np.min(test_img_array))

        test_images.append(norm_test_img)

    f.close()


def load_train_data(file_path):
    f = open(file_path, "r")

    f.readline()
    for train_img_data in f:
        train_img, train_img_label = train_img_data.split(",")
        train_img_array = cv2.imread(os.path.join(train_and_validation_dir, train_img), cv2.IMREAD_GRAYSCALE)
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
        validation_img_array = cv2.imread(os.path.join(train_and_validation_dir, validation_img), cv2.IMREAD_GRAYSCALE)
        # Normalizam imaginea
        norm_validation_img = (validation_img_array - np.min(validation_img_array)) / (np.max(validation_img_array) - np.min(validation_img_array))

        validation_images.append(norm_validation_img)
        validation_labels.append(int(validation_img_label))

    f.close()
##########################################################################################



# Incarcarea datelor din fisiere/foldere si pregatirea acestora pentru a putea fi introduse in model
##########################################################################################
load_test_data(test_metadata_file)
load_train_data(train_metadata_file)
load_validation_data(validation_metadata_file)

test_images = np.array(test_images)
# Pentru a introduce datele in model, trebuie sa transformam dimensiunea array-ului din 3D in 2D
nr_images, width, height = test_images.shape
test_images = test_images.reshape((nr_images, width * height))

train_images = np.array(train_images)
# Pentru a introduce datele in model, trebuie sa transformam dimensiunea array-ului din 3D in 2D
nr_images, width, height = train_images.shape
train_images = train_images.reshape((nr_images, width * height))
train_labels = np.array(train_labels)

validation_images = np.array(validation_images)
# Pentru a introduce datele in model, trebuie sa transformam dimensiunea array-ului din 3D in 2D
nr_images, width, height = validation_images.shape
validation_images = validation_images.reshape((nr_images, width * height))
validation_labels = np.array(validation_labels)
##########################################################################################



# Modelul SVM
##########################################################################################
# Configuram modelul
SVM = svm.SVC(C=25, kernel="linear")

# Antrenam modelul
SVM.fit(X=train_images, y=train_labels)

# Obtinem label-urile prezise de catre model
predicted_labels.extend(SVM.predict(X=validation_images))
for pred in range(len(predicted_labels)):
    predicted_labels[pred] = int(predicted_labels[pred])

# Populam fisierul aferent predictiilor
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

# Determinam acuratetea modelului
SVM_accuracy = accuracy_score(y_true=validation_labels, y_pred=predicted_labels)
print(f"\nSVM accuracy: {SVM_accuracy}\n")

# Obtinem matricea de confuzie pentru predictiile generate de catre clasificator
SVM_cm = confusion_matrix(y_true=validation_labels, y_pred=predicted_labels)
print(f"SVM confusion matrix:\n{SVM_cm}")
##########################################################################################