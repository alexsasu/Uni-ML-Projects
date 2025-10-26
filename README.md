# Uni-ML-Projects

"Image-Sentence Pair Matching" was an individual project made during the Deep Learning course taken in the 2nd year of the Artificial Intelligence master program at the Faculty of Mathematics and Computer Science, University of Bucharest.

"Tackling Unsupervised Learning" and "Sentence Pair Classification" were individual projects made during the Practical Machine Learning course taken in the 1st year of the Artificial Intelligence master program at the Faculty of Mathematics and Computer Science, University of Bucharest.

"Deep Hallucination Classification" was an individual project made during the Artificial Intelligence course taken in the 2nd year of the Computer Science bachelor program at the Faculty of Mathematics and Computer Science, University of Bucharest.

The documentation and implementation details of each project can be consulted in the associated directory.
<!--More details about each project, such as results, models architecture, and code, can be consulted in their associated directory.-->

## Image-Sentence Pair Matching

This project consisted of participating in a [Kaggle competition](https://www.kaggle.com/competitions/isp-match-dl-2024/) for determining whether pairs of images and English texts matched or not (binary classification). We were required to use **deep learning models**. The main models used were a **CNN** and an **LSTM**, and they were used either as subnets in a bigger model, or individually. Basic **preprocessing methods** were employed, namely: text to lowercase, punctuation removal, stop words removal, image normalization. Features were extracted with the help of a **Tensorflow Tokenizer**. Due to a lack of comprehensiveness in the train/validation sets, and a biased test dataset with only 15 unique, more descriptive texts than train/validation, poor test performance was achieved.

<details>
<summary><b>Results:</b></summary>

Scores represent classification accuracies.
| # | Strategy (check documentation) | Private leaderboard (70% test) | Validation |
|:---:|---|---:|---:|
| 1 | Leaderboard best | 69.56% | - |
|...|...|...|...|
| - | Leaderboard baseline | 52.50%| - |
| 44 | Strategy 4: Strategy 1 Revisited | 51.12% | ~66% |
| - | Strategy 1: CNN and LSTM + TF Tokenizer + basic preprocessing | 50.81% | ~67% |
| - | Strategy 3: Strategy 1 w/o last layer + Random Forest | 50.43% | ~68% |
| - | Strategy 2: LSTM + TF Tokenizer + basic preprocessing | 50.43% | ~67% |
|...|...|...|...|
| 68 | Leaderboard worst | 47.56% | - |

</details>

## Tackling Unsupervised Learning

The main task of this project was to use two **clustering methods** on a labeled dataset. The chosen dataset was [Asirra](https://huggingface.co/datasets/cats_vs_dogs), containing pictures of cats and dogs, and the chosen models were **K-means** and **DBSCAN**. Methods such as the **elbow** and **silhouette score** were used in order to determine the appropriate number of clusters, and after clustering the images, the **Hungarian algorithm** was used to realize a mapping between cluster labels and actual class labels. The **feature extraction** method used consisted of the following: bringing each pixel of the image in the range [0.0, 1.0] through PyTorch’s ToTensor function, processing the image using the formula output[channel] = (input[channel] − mean[channel]) / std[channel] with the help of the PyTorch function Normalize, transforming the image from a 3D array into a 1D array of length height * width * channels.

## Sentence Pair Classification

For this project we had to take part in a [Kaggle competition](https://www.kaggle.com/competitions/sentence-pair-classification-pml-2023/) for classifying pairs of Romanian sentences (multiclass classification). Multiple machine learning models were tested: **fastText**, **LSTM**, **SVM**, **Random Forest**; along with multiple feature extraction methods: **keras Tokenizer**, **TF-IDF**, **Word2Vec skip-gram embeddings**; and multiple **preprocessing methods**: text to lowercase, punctuation removal, stop words removal. The datasets were imbalanced, and in order to address this issue, **balanced weights** were used for the samples.

<details>
<summary><b>Results:</b></summary>

Scores represent macro F1 values.
| # | Strategy (check documentation) | Private leaderboard (70% test) | Validation |
|:---:|---|---:|---:|
| 1 | Leaderboard best | 65.76% | - |
|...|...|...|...|
| 22 | Strategy 6: Strategy 4 + added validation set to train set | 65.11% | 100.00% |
| - | Strategy 4: Random Forest + TF-IDF + basic preprocessing | 63.75% | ~34% |
| - | Strategy 3: SVM + TF-IDF + basic preprocessing | 58.39% | ~40% |
| - | Strategy 2: LSTM + Keras Tokenizer + basic preprocessing | 52.05% | ~39% |
| - | Strategy 1: fastText + basic preprocessing | 45.27% | ~36% |
| - | Strategy 5: LSTM + Word2Vec + Keras Tokenizer + basic preprocessing | - | ~39% |
| - | Leaderboard baseline | 19.32%| - |
|...|...|...|...|
| 115 | Leaderboard worst | 8.62% | - |

</details>

## Deep Hallucination Classification

The project required joining a [Kaggle competition](https://www.kaggle.com/competitions/unibuc-2022-s24/) for classifying images hallucinated by **deep generative models** (multiclass classification). A multitude of models were tried: **CNN**, **Random Forest**, **KNN**, **SVM**; but the model that was mainly used was a CNN. By using the model architecture present in the "Deep Hallucination Classification" directory, the **5th place out of 128 participants** was attained.

<details>
<summary><b>Results:</b></summary>

Scores represent classification accuracies.
| # | Strategy (check documentation) | Private leaderboard (75% test) | Validation |
|:---:|---|---:|---:|
| 1 | Leaderboard best | 69.69% | - |
|...|...|...|...|
| 5 | CNN | 64.91% | ~65% |
| - | Random Forest and Grid Search CV | - | ~43% |
| - | SVM | - | ~39% |
| - | KNN for regression | - | ~26% |
| - | Leaderboard baseline | 15.27%| - |
|...|...|...|...|
| 128 | Leaderboard worst | 14.32% | - |

</details>
