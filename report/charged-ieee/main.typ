#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [Aerial Imagery - Challenge 1 AML],
  abstract: [
    This report presents a comparative analysis of different machine learning models developed to classify aerial images containing a specific type of cactus (Neobuxbaumia tetetzo). The goal of the project is to support biodiversity monitoring efforts such as the VIGIA project in Mexico, by identifying the presence of cacti in 32x32 aerial image patches. We describe the dataset, preprocessing methods, and the models trained so far. Final performance comparison and model selection for test set classification will follow upon completion of all experiments.
  ],
  authors: (
    (
      name: "Alex Argese",
      department: [Student],
      organization: [EURECOM],
      location: [Biot, France],
      email: "alex.argese@eurecom.fr"
    ),
    (
      name: "Cristian Degni",
      department: [Student],
      organization: [EURECOM],
      location: [Biot, France],
      email: "cristian.degni@eurecom.fr"
    ),
    (
      name: "Miriam Lamari",
      department: [Student],
      organization: [EURECOM],
      location: [Biot, France],
      email: "miriam.lamari@eurecom.fr"
    ),
    (
      name: "Enrico Sbuttoni",
      department: [Student],
      organization: [EURECOM],
      location: [Biot, France],
      email: "enrico.sbuttoni@eurecom.fr"
    ),
  ),
  index-terms: ("Machine Learning", "Cactus Detection", "Aerial Imagery", "CNN", "ResNet"),
)

= Introduction
Recognizing flora from aerial imagery plays a key role in conservation and ecological monitoring. The VIGIA project aims to automate surveillance of protected areas in Mexico, starting with detecting *Neobuxbaumia tetetzo* cacti from aerial photographs. In this challenge, we develop machine learning models to distinguish between image patches that do and do not contain a cactus.

The dataset consists of labeled 32×32 pixel RGB images, and the task is formulated as a binary classification problem. We explored several algorithms: Logistic Regression, Support Vector Machines (SVM), a custom Convolutional Neural Network (CNN), and a pretrained ResNet model. All models are trained on a split of the original dataset, with a portion held out for validation and testing. We are currently finalizing the training and evaluation stages, and will report the best-performing model on the test set.

= Dataset Analysis and Preprocessing
The training set consists of 17,500 labeled images (balanced across classes), provided in the `train/` directory, with labels in `train.csv`. Each image corresponds to a 32×32 RGB aerial photo.

We began by loading and analyzing the dataset:
- The dataset appears relatively balanced: both classes (cactus / no cactus) are comparably represented.
- Images were normalized (pixel values scaled to [0,1]).
- A validation split of 20% was created from the training set to evaluate generalization.
- Data augmentation (random flips, rotations) was applied for training CNN-based models to increase robustness.

= Models Trained

== Logistic Regression
A simple baseline model using pixel intensities as flattened input features. As expected, the performance is modest due to the model’s linear nature and lack of spatial context.

== Support Vector Machine
An SVM with RBF kernel was used after flattening and scaling the input images. It performed slightly better than logistic regression but is limited by computational cost and lack of convolutional context.

== Convolutional Neural Network (CNN)
We implemented a custom CNN composed of multiple convolutional layers followed by max-pooling and dense layers. The model achieved promising accuracy on the validation set thanks to its ability to learn spatial patterns specific to the cactus shapes and textures.

== ResNet (Pretrained)
We employed a transfer learning approach using a ResNet model pretrained on ImageNet. The architecture was modified to accept 32×32 images, and only the final layers were fine-tuned. This model has shown strong preliminary results.

= Next Steps
At the time of writing, the training of all models is nearly complete. Once finished, we will compare their validation accuracies and use the best-performing model to predict labels on the test dataset. These predictions will then be reported and analyzed.

= References
This section will include all references used, once the analysis is finalized and citations are added.
