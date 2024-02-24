# Imports

from collections import Counter
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_samples, silhouette_score
from sklearn.model_selection import GridSearchCV
import torchvision.transforms as transforms

# --------------------------------------------------

# Loading the samples

# The folder that contains the actual images
samples_dir = "cats_and_dogs"

# Files that contain the names of images and their labels
train_metadata_file = "train_metadata_1.6k.txt"
val_metadata_file = "val_metadata_1.6k.txt"
test_metadata_file = "test_metadata_1.6k.txt"

# Helper function for loading the samples
def load_samples(dir, file_path):
    data, labels = [], []

    with open(file_path) as f:
        lines = [line.rstrip() for line in f]
        for img_info in lines[1:]:
            img_name, label = img_info.split(',')
            img = cv2.imread(os.path.join(dir, img_name))
            img = cv2.resize(img, (200, 200)) # giving each image the same dimensions so that the models don't raise errors
            
            data.append(img)
            labels.append(int(label))
    
    return data, labels

# Loading the samples through the helper function
train_data, train_labels = load_samples(samples_dir, train_metadata_file)
val_data, val_labels = load_samples(samples_dir, val_metadata_file)
test_data, test_labels = load_samples(samples_dir, test_metadata_file)

label_types = [0, 1]

# --------------------------------------------------

# Function for extracting features from images
# Feature extraction method: image to tensor + normalization from PyTorch + 1D array 
# of size height * width * channels

def feature_extraction_method(data):
    new_data = []

    for img in data:
        transformer = transforms.Compose([transforms.ToTensor()])
        img_tensor = transformer(img)
        mean, std = img_tensor.mean([1, 2]), img_tensor.std([1, 2])

        transformer_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        img = transformer_norm(img)

        img = np.array(img)
        new_data.append(img)
    
    new_data = np.asarray(new_data)
    no_samples = new_data.shape[0]
    new_data = new_data.reshape((no_samples, -1)) # for transforming each image into a 1D array of pixels
    
    return new_data

# --------------------------------------------------

# Extracting the features from data

train_data = feature_extraction_method(train_data)
train_labels = np.asarray(train_labels)
val_data = feature_extraction_method(val_data)
val_labels = np.asarray(val_labels)
test_data = feature_extraction_method(test_data)
test_labels = np.asarray(test_labels)

# --------------------------------------------------

# Function for generating the predicted class labels, based on the cluster labels

def get_class_labels_pred(true_labels, cluster_labels):
    cm = confusion_matrix(true_labels, cluster_labels)
    association_matrix = 1 / cm
    row_ind, col_ind = linear_sum_assignment(association_matrix)
    labels_assignment = dict(zip(row_ind, col_ind))
    class_labels_pred = [labels_assignment[cluster_label] for cluster_label in cluster_labels]
    class_labels_pred = np.asarray(class_labels_pred)

    return class_labels_pred

# --------------------------------------------------

# Code taken from laboratory no. 5 for the visualisation of the clustering ability
# of the model

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray']
MARKERS = ['o', 'v', 's', '<', '>', '8', '^', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

def plot2d(X, y_pred, y_true, mode=None, mode_str=None, centroids=None):
    transformer = None
    X_r = X
    
    if mode is not None:
        transformer = mode(n_components=2)
        X_r = transformer.fit_transform(X)

    assert X_r.shape[1] == 2, 'plot2d only works with 2-dimensional data'


    plt.grid()
    for ix, iyp, iyt in zip(X_r, y_pred, y_true):
        plt.plot(ix[0], ix[1], 
                    c=COLORS[iyp], 
                    marker=MARKERS[0])
        
    if centroids is not None:
        C_r = centroids
        if transformer is not None:
            C_r = transformer.fit_transform(centroids)
        for cx in C_r:
            plt.plot(cx[0], cx[1], 
                        marker=MARKERS[-1], 
                        markersize=10,
                        c='red')
    
    if mode_str == "PCA":
        plt.title("Features clustering visualised with PCA")
    else:
        plt.title("Features clustering visualised with TSNE")
    plt.xlabel("x coord")
    plt.ylabel("y coord")

    plt.show()

def plot3d(X, y_pred, y_true, mode=None, mode_str=None, centroids=None):
    transformer = None
    X_r = X
    if mode is not None:
        transformer = mode(n_components=3)
        X_r = transformer.fit_transform(X)

    assert X_r.shape[1] == 3, 'plot2d only works with 3-dimensional data'

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.elev = 30
    ax.azim = 120

    for ix, iyp, iyt in zip(X_r, y_pred, y_true):
        ax.plot(xs=[ix[0]], ys=[ix[1]], zs=[ix[2]], zdir='z',
                    c=COLORS[iyp], 
                    marker=MARKERS[0])
        
    if centroids is not None:
        C_r = centroids
        if transformer is not None:
            C_r = transformer.fit_transform(centroids)
        for cx in C_r:
            ax.plot(xs=[cx[0]], ys=[cx[1]], zs=[cx[2]], zdir='z',
                        marker=MARKERS[-1], 
                        markersize=10,
                        c='red')
            
    if mode_str == "PCA":
        plt.title("Features clustering visualised with PCA")
    else:
        plt.title("Features clustering visualised with TSNE")
            
    plt.show()

# --------------------------------------------------

# Searching for appropriate parameters for the DBSCAN model

# Helper function for getting the number of data points from each cluster

def get_data_points_info(cluster_labels):
    items_count = Counter(cluster_labels)

    info = dict()
    info['total'] = len([label for label in cluster_labels if label != -1])
    info.update(dict(items_count))

    return info

# Performing a grid search-like approach, in order to find appropriate parameters
# for the model, based on the clustering ability, silhouette scores, and accuracies

eps_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
min_samples_arr = [i for i in range(2, 11)]

data_points_info, scores, params, accs = [], [], [], []

for eps_opt in eps_arr:
    for min_samples_opt in min_samples_arr:
        dbscan = DBSCAN(metric='cosine')
        dbscan.set_params(eps=eps_opt, min_samples=min_samples_opt)
        train_cluster_labels = dbscan.fit_predict(train_data)

        train_data_used = train_data[train_cluster_labels != -1]
        train_labels_used = train_labels[train_cluster_labels != -1]
        train_cluster_labels_used = train_cluster_labels[train_cluster_labels != -1]
        if len(np.unique(train_cluster_labels_used)) == len(np.unique(train_labels_used)) and len(np.unique(train_cluster_labels_used)) >= 2:
            print(np.unique(train_labels_used))
            print(np.unique(train_cluster_labels_used))
            print()
            train_labels_pred = get_class_labels_pred(train_labels_used, train_cluster_labels_used)

        data_points_info.append(get_data_points_info(train_cluster_labels))
        if len(np.unique(train_cluster_labels_used)) == len(np.unique(train_labels_used)) and len(np.unique(train_cluster_labels_used)) >= 2:
            scores.append(silhouette_score(train_data_used, train_cluster_labels_used))
        else:
            scores.append(-2)
        params.append(dbscan.get_params())
        if len(np.unique(train_cluster_labels_used)) == len(np.unique(train_labels_used)) and len(np.unique(train_cluster_labels_used)) >= 2:
            accs.append(accuracy_score(train_labels_used, train_labels_pred))
        else:
            accs.append(-1)

print(params)
print()
for i in range(len(eps_arr)):
    for j in range(len(min_samples_arr)):
        print(data_points_info[9 * i + j], f" | [{9 * i + j}, {eps_arr[i]}, {min_samples_arr[j]}, {scores[9 * i + j]}, {accs[9 * i + j]}]")

# --------------------------------------------------

# Training the model with the choice of parameters that was deemed appropriate

dbscan = DBSCAN(eps=0.4, min_samples=9, metric='cosine')
train_cluster_labels = dbscan.fit_predict(train_data)

# --------------------------------------------------

# Generating the predicted train labels

train_data_used = train_data[train_cluster_labels != -1]
train_labels_used = train_labels[train_cluster_labels != -1]
train_cluster_labels_used = train_cluster_labels[train_cluster_labels != -1]

train_labels_pred = get_class_labels_pred(train_labels_used, train_cluster_labels_used)

# --------------------------------------------------

# Visualising the clustering ability of the model

plot2d(train_data, train_cluster_labels, train_labels, mode=PCA, mode_str="PCA")

# --------------------------------------------------

# Code taken from scikit-learn for visualising the clusters silhouette score plot of the model

score = silhouette_score(train_data, train_cluster_labels)

# Compute the silhouette scores for each sample
images_silhouette = silhouette_samples(train_data_used, train_cluster_labels_used)

plot_y_lower = 10
for i in range(len(np.unique(train_cluster_labels_used))):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    cluster_samples_silhouette_scores = images_silhouette[train_cluster_labels_used == i]
    cluster_samples_silhouette_scores.sort()

    cluster_size = cluster_samples_silhouette_scores.shape[0]
    plot_y_upper = plot_y_lower + cluster_size

    color = cm.nipy_spectral(float(i) / len(np.unique(train_cluster_labels_used)))
    plt.fill_betweenx(
        np.arange(plot_y_lower, plot_y_upper),
        0,
        cluster_samples_silhouette_scores,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    plt.text(-0.05, plot_y_lower + 0.5 * cluster_size, str(i))

    # Compute the new plot_y_lower for next plot
    plot_y_lower = plot_y_upper + 10  # 10 for the 0 samples

plt.title("Clusters silhouette score plot")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
plt.axvline(x=score, color="red", linestyle="--")

plt.yticks([]) # Clear the yaxis labels / ticks
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()

# --------------------------------------------------

# Showing results on the test data

print(classification_report(train_labels_used, train_labels_pred, labels=label_types))
print()
cm = confusion_matrix(train_labels_used, train_labels_pred)
print(cm)
sns.heatmap(cm, annot=True)
plt.ylabel('True', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

# --------------------------------------------------

# Generating the random chance, for comparison with the model's accuracy

dummy = DummyClassifier()
dummy.fit(train_data, train_labels)
print('Train acc: ', dummy.score(train_data_used, train_labels_used))

# --------------------------------------------------

# Training and testing the Random Forest classifier, to obtain the supervised baseline

rf = RandomForestClassifier()
rf.fit(train_data_used, train_labels_used)

train_labels_pred = rf.predict(train_data_used)

print(classification_report(train_labels_used, train_labels_pred, labels=label_types))
print()
cm = confusion_matrix(train_labels_used, train_labels_pred)
print(cm)
sns.heatmap(cm, annot=True)
plt.ylabel('True', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

# --------------------------------------------------
