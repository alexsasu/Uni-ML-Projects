# Uni-ML-Projects

The "Sentence Pair Classification" and "Tackling Unsupervised Learning" projects were made during the Practical Machine Learning course taken in the 1st year of the Artificial Intelligence master program at the Faculty of Mathematics and Computer Science, University of Bucharest.

The "Deep Hallucination Classification" project was made during the Artificial Intelligence course taken in the 2nd year of the Computer Science bachelor program at the Faculty of Mathematics and Computer Science, University of Bucharest.

More details about each project, such as results, models architecture, and code, can be consulted in their associated directory.

## Sentence Pair Classification

This project consisted of participating in a [Kaggle competition](https://www.kaggle.com/competitions/sentence-pair-classification-pml-2023/) for an NLP task on a Romanian corpus. Multiple machine learning models were tested: **fastText**, **LSTM**, **SVM**, **Random Forest**; along with multiple feature extraction methods: **keras Tokenizer**, **TF-IDF**, **Word2Vec skip-gram embeddings**; and multiple preprocessing methods: **text to lowercase**, **punctuation removal**, **stop words removal**. The datasets were imbalanced, and in order to address this issue, **balanced weights** were used for the samples.

## Tackling Unsupervised Learning

This project consisted of using two clustering methods on a labeled dataset. The dataset chosen was one with pictures of cats and dogs ([Asirra](https://huggingface.co/datasets/cats_vs_dogs)), and the models were **K-means** and **DBSCAN**. Methods such as the **elbow** and **silhouette score** were used in order to determine the appropriate number of clusters, and after clustering the images, the **Hungarian algorithm** was used in order to realize a mapping between cluster labels and actual class labels. The feature extraction method used consisted of the following: bringing each pixel of the image in the range [0.0, 1.0] through **PyTorch’s ToTensor function**, processing the image using the formula **output[channel] = (input[channel] − mean[channel]) / std[channel]** with the help of the **PyTorch function Normalize**, transforming the image from a 3D array into a 1D array of length height * width * channels.

## Deep Hallucination Classification

This project consisted of participating in a [Kaggle competition](https://www.kaggle.com/competitions/unibuc-2022-s24/) for classifying images hallucinated by deep generative models. A multitude of models were tried: **CNN**, **Random Forest**, **KNN**, **SVM**; but the model that was mainly used was a custom-made **CNN**. By using the model architecture present in the "Deep Hallucination Classification" directory, the **5th place out of 128 participants** was attained.
