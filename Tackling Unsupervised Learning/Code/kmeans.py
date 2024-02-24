# Imports

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from sklearn.cluster import KMeans
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

# Searching for appropriate parameters for the K-means model

# Searching for the appropriate amount of clusters

n_clusters = [i for i in range(2, 13)]

silhouette_scores = []

for clusters in n_clusters:
    kmeans = KMeans(n_init='auto')
    kmeans.set_params(n_clusters=clusters)
    kmeans.fit(train_data)
    silhouette_scores.append(silhouette_score(train_data, kmeans.labels_))

print(silhouette_scores)

x = [i for i in range(2, 13)]
y = silhouette_scores

plt.plot(x, y)
plt.xlabel('Clusters')
plt.ylabel('Silhouette score')
plt.show()

# Performing grid search

kmeans_grid = KMeans(n_clusters=2, n_init='auto')
grid_search_parameters = {
                'init': ('k-means++', 'random'),
                'algorithm': ('lloyd', 'elkan')
            }

grid_search = GridSearchCV(kmeans_grid, grid_search_parameters)
grid_search.fit(train_data)

print(grid_search.best_params_)

# Performing a grid search-like approach, in order to find the best accuracy and
# the model parameters associated with it

init_arr = ['k-means++', 'random']
algorithm_arr = ['lloyd', 'elkan']

scores, params, accs = [], [], []

for init_opt in init_arr:
    for algorithm_opt in algorithm_arr:
        kmeans = KMeans(n_clusters=2, init=init_opt, n_init='auto', algorithm=algorithm_opt)
        kmeans.set_params(init=init_opt, algorithm=algorithm_opt)
        kmeans.fit(train_data)
        train_cluster_labels = kmeans.labels_
        train_labels_pred = get_class_labels_pred(train_labels, train_cluster_labels)

        val_cluster_labels = kmeans.predict(val_data)
        val_labels_pred = get_class_labels_pred(val_labels, val_cluster_labels)
        
        scores.append((kmeans.inertia_, silhouette_score(train_data, train_cluster_labels)))
        params.append(kmeans.get_params())
        accs.append((accuracy_score(train_labels, train_labels_pred), accuracy_score(val_labels, val_labels_pred)))

print(scores)
print()
print(params)
print()
print(accs)

# --------------------------------------------------

# Training the model with the choice of parameters that was deemed appropriate

kmeans = KMeans(n_clusters=2, init='k-means++', n_init='auto', algorithm='elkan')
kmeans.fit(train_data)

# --------------------------------------------------

# Using the model to generate predictions

train_cluster_labels = kmeans.labels_
train_labels_pred = get_class_labels_pred(train_labels, train_cluster_labels)
val_cluster_labels = kmeans.predict(val_data)
val_labels_pred = get_class_labels_pred(val_labels, val_cluster_labels)
test_cluster_labels = kmeans.predict(test_data)
test_labels_pred = get_class_labels_pred(test_labels, test_cluster_labels)

# --------------------------------------------------

# Visualising the clustering ability of the model

plot2d(train_data, train_cluster_labels, train_labels, mode=PCA, mode_str="PCA", centroids=kmeans.cluster_centers_)
# plot2d(test_data, test_cluster_labels, test_labels, mode=PCA, mode_str="PCA")

# --------------------------------------------------

# Code taken from scikit-learn for visualising the clusters silhouette score plot of the model

score = silhouette_score(train_data, train_cluster_labels)

# Compute the silhouette scores for each sample
images_silhouette = silhouette_samples(train_data, train_cluster_labels)

plot_y_lower = 10
for i in range(len(np.unique(train_cluster_labels))):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    cluster_samples_silhouette_scores = images_silhouette[train_cluster_labels == i]
    cluster_samples_silhouette_scores.sort()

    cluster_size = cluster_samples_silhouette_scores.shape[0]
    plot_y_upper = plot_y_lower + cluster_size

    color = cm.nipy_spectral(float(i) / len(np.unique(train_cluster_labels)))
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

print(classification_report(test_labels, test_labels_pred, labels=label_types))
print()
cm = confusion_matrix(test_labels, test_labels_pred)
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
print('Train acc: ', dummy.score(train_data, train_labels))
print('Validation acc:', dummy.score(val_data, val_labels))
print('Test acc:', dummy.score(test_data, test_labels))

# --------------------------------------------------

# Training and testing the Random Forest classifier, to obtain the supervised baseline

rf = RandomForestClassifier()
rf.fit(train_data, train_labels)

train_labels_pred = rf.predict(train_data)
val_labels_pred = rf.predict(val_data)
test_labels_pred = rf.predict(test_data)

print(classification_report(test_labels, test_labels_pred, labels=label_types))
print()
cm = confusion_matrix(test_labels, test_labels_pred)
print(cm)
sns.heatmap(cm, annot=True)
plt.ylabel('True', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

# --------------------------------------------------
