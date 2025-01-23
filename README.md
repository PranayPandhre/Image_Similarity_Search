#  Image Similarity Search using FMNIST dataset

This repository contains my experiments on **Performing image similarity search** using **Autoencoders**, **ResNet101**, **Siamese_Network**, and **SIFT**. The experiments were conducted on the **Fashion MNIST (FMNIST)** dataset which contains grayscale fashion clothing images. The repository includes the experiemnts conducted using the mentioned methods, and the code documentation (which includes the results of the experiemnts) of each experiment in it's respective folder.

---

## Table of Contents
- **[Introduction](#introduction)**
- **[Dataset](#dataset)**
- **[Repository Structure](#repository-structure)**
- **[Methods Used](#methods-used)**
- **[Results_Analysis](#results_analysis)**
- **[References](#references)**
- **[Requirements](#requirements)**

---

## Introduction

The methodologies used for performing **Image similarity search** are as follows:

- **Autoencoders**
- **ResNet101** (Deeper CNN model for better feature extraction compared to commonly used ResNet50
- **Siamese Networks** for performing image similarity.
- **SIFT** mathematical non-machine learning approach for performing image similarity.

---

## Dataset

**FMNIST:**
    - A dataset containing grayscale fashion clothing images with the following categories.
    - Includes T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots. 
    - Size of images in the dataset 28x28 pixels.
    - Lightweight and commonly used as a benchmark image classification and similarity search methods.

---

## Repository Structure

```plaintext

Autoencoders/
├── Autoencoder.ipynb
├── Autoencoder_documentation.pdf
├── autoencoder_model.h5
ResNet101/
├── Resnet101.ipynb
├── Resnet_documentation.pdf
├── Siamese_network
    ├── Siamese_network.ipynb
    ├── Siamese_Network_documentation.pdf
    ├── Siamese_saved_models/
        ├── siamese_model.h5
        └── siamese_model.keras
SIFT/
├── SIFT.ipynb
├── SIFT_documentation.pdf

Scores/
│   ├── (Screenshots of All approaches model scores)
│
README.md
```

## Methods Used

### **1. Autoencoder**
An Autoencoder is a type of neural network designed for unsupervised learning that compresses input data into a lower-dimensional representation (latent space) and reconstructs it back to the original dimensions. It is widely used for image similarity search because the latent representations learned by the encoder capture essential features of the input image, discarding noise and irrelevant details. These latent vectors are used to compute similarities (e.g., cosine similarity or Euclidean distance) between images efficiently.

#### **Process**
- Trained a convolutional autoencoder on the dataset.
- This autoencoder uses convolutional layers (Conv2D) for feature extraction and representation learning, making it suitable for image data.
- It is designed to handle spatial relationships in images (height, width, and channels), unlike traditional dense autoencoders that flatten input data.

#### **Advantages**
- They are computationally less intensive compared to deep CNNs and can be tuned for efficient encoding-decoding operations.
- Autoencoders are scalable and can handle large datasets with high-dimensional features.

---

#### **2. ResNet101**
ResNet (Residual Network) is a deep convolutional neural network that revolutionized deep learning by introducing residual connections, which mitigate the vanishing gradient problem, enabling the effective training of very deep networks. It is widely used for image similarity search because its pre-trained versions (e.g., ResNet-50, ResNet-101) can extract rich, high-dimensional feature embeddings from images, capturing complex patterns and structures. These embeddings are particularly suitable for measuring image similarity using metrics like cosine distance. 

#### **Process**
- ResNet101 leverages residual connections (shortcut connections) to allow gradients to flow more easily through the network during backpropagation.
- It consists of 101 layers, including convolutional layers and batch normalization, which work together to extract hierarchical features from input images.
- Features from the final or intermediate layers of ResNet101 are used as high-dimensional feature vectors for similarity computation (cosine similarity).

#### **Advantages**
- ResNet101 is a deeper CNN compared to ResNet50 and this helps in better processing and feature extraction
- The learned feature representations are highly robust and generalizable, making ResNet101 an excellent choice for tasks like feature extraction.
- Simple and effective for smaller datasets

---

#### **3. Siamese Network**
- A Siamese Network is a neural network architecture specifically designed for similarity learning, where the goal is to determine whether two inputs are similar or dissimilar. It consists of two identical subnetworks that process two inputs independently and produce feature embeddings, which are then compared using a distance metric (e.g., Euclidean distance) to quantify similarity. Siamese Networks are widely used for image similarity search because they are explicitly trained to learn a similarity function rather than classifying inputs.

#### **Process**
- The dataset is prepared using the **create_triplets** function creates triplets of images (anchor, positive, and negative) to train the network. Positive images belong to the same class as the anchor, while negative images belong to a different class.
- The **build_siamese_network** function constructs a feature extraction model using convolutional layers to generate a compact 64-dimensional embedding for each image.
- The custom **triplet_loss** ensures embeddings for anchor-positive pairs are closer than anchor-negative pairs by a specified margin (0.2), improving similarity learning.
- The **create_triplet_model** function combines the Siamese network with three inputs (anchor, positive, and negative) and stacks their embeddings for loss computation during training.
- The **train_model** function trains the triplet model on the generated triplets using the Adam optimizer, monitoring performance on a validation set to ensure effective learning.

#### **Advantages**
- Siamese networks are designed to learn whether two inputs are similar, making them ideal for tasks such as image matching.
- After training, only the feature extractor (base model) is needed for embeddings. Comparing embeddings (e.g., via Euclidean distance) is computationally cheaper than training or testing traditional classification models for new classes.
- In retrieval tasks (e.g., finding similar images), Siamese networks eliminate the need for exhaustive search by mapping inputs into an embedding space where simple distance metrics suffice for comparison.

---

#### **4. SIFT**
- SIFT method is widely used for image similarity search due to its robustness and ability to identify and match distinctive features in images, regardless of variations. SIFT methodology mainly involves keypoints and descriptors. Keypoints and descriptors  are fundamental concepts used to identify and describe distinctive regions in an image for tasks like similarity search. Keypoints identify where the distinctive features are located in the image. Each keypoint corresponds to a specific pixel in the image, typically representing corners, edges, or blobs that are rich in texture and less likely to change under transformations. Descriptors describe what those features look like. SIFT generates a 128-dimensional vector for each keypoint. This vector encodes gradient magnitudes and directions in a local region around the keypoint.

#### **Process**
- The SIFT process identifies keypoints (scale-invariant points of interest) and computes feature descriptors for images. Descriptors from two images are matched using a brute-force matcher with Lowe's ratio test to filter good matches.
- A similarity score is calculated based on the number of valid matches relative to the number of keypoints. Matches are visualized using OpenCV, highlighting the corresponding features between images.
- The process compares representative images from Fashion MNIST classes with random samples to evaluate intra-class and inter-class similarity, plotting matches and reporting similarity scores for analysis.

#### **Advantages**
- SIFT is robust to changes in scale and orientation, ensuring reliable feature detection and matching across transformed images.
- The 128-dimensional descriptors provide detailed representations of local image regions, allowing accurate and discriminative matching.
- SIFT can identify matches even when images are cropped or contain partial overlaps, making it ideal for image similarity search.

   
## Results_Analysis


## References

- [Autoencoders and Dimensionality Reduction](https://arxiv.org/abs/1512.03385) - _A key paper that describes the use of autoencoders for reducing dimensionality and unsupervised learning._
- [Explainable Image Similarity: Integrating Siamese Networks and Grad-CAM](https://arxiv.org/pdf/2310.07678) 
- [Siamese Networks for One-Shot Learning](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) - _A foundational paper introducing Siamese Networks for tasks requiring one-shot learning._
- [Transfer Learning with ResNet](https://arxiv.org/abs/1512.03385) - _An essential reference on ResNet architecture and its applications in transfer learning._
- [FMNIST Dataset Overview](https://github.com/zalandoresearch/fashion-mnist) - _Detailed information on the FMNIST dataset, including benchmarks and evaluation metrics._

## Requirements
