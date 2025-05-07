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
Monitoring biodiversity is a growing priority in the context of climate change and human-driven land transformation. In this challenge, we focus on detecting *Neobuxbaumia tetetzo*, a columnar cactus species, in 32×32 aerial images using machine learning. The dataset was derived from the VIGIA project in Mexico.

= Dataset and Preprocessing
The dataset consists of 17,500 labeled 32×32 RGB images in the `train/` folder, and a set of unlabeled test images. Each image is associated with a binary label (`has_cactus = 0` or `1`).

The data was split into:
- 80% training
- 10% validation
- 10% test (manual hold-out)

Images were normalized and resized if necessary. For CNN-based models, we applied data augmentation (horizontal/vertical flip, rotation).

= Models Evaluated

== Logistic Regression
Baseline model using flattened pixel values. Its performance is limited due to the absence of spatial pattern learning.

- *F1 Score*: `0.8098`
- *Accuracy*: `83.4%`

== Support Vector Machine
SVM with RBF kernel and grid search on hyperparameters (C, gamma). Performed better than logistic regression, but with higher computation time.

- *F1 Score*: `0.8571`
- *Accuracy*: `86.2%`

== Convolutional Neural Network (CNN)
Custom CNN with two convolutional layers, batch normalization, ReLU, and dropout. Trained for 10 epochs with data augmentation.

- *F1 Score*: `0.9554`
- *Accuracy*: `96.0%`

== ResNet18
Transfer learning with PyTorch’s ResNet18 pretrained on ImageNet. Only the final layer was fine-tuned.

- *F1 Score*: `0.9776`
- *Accuracy*: `98.0%`

= Metric Justification
Since the dataset is slightly imbalanced, we adopted *F1 score* as the primary evaluation metric. It balances precision and recall, which is crucial in ecological monitoring where false negatives (missed cacti) may have high cost.

= Results Summary

#figure(
  caption: [Performance Comparison of Models on the Validation Set],
  placement: top,
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    table.header[Model][F1 Score][Accuracy],
    [Logistic Regression], [0.8098], [83.4%],
    [SVM], [0.8571], [86.2%],
    [CNN], [0.9554], [96.0%],
    [ResNet18], [0.9776], [98.0%],
  )
) <tab:results>

= Conclusion and Next Steps
Among the four models tested, *ResNet18* demonstrated the best performance in both accuracy and F1 score. It was therefore selected to generate predictions on the unlabeled test set. In future work, further improvement could involve model ensembling or unsupervised pretraining on aerial imagery.

= References
This report is inspired by the VIGIA project as described in:

Efren López-Jiménez et al., *Columnar Cactus Recognition in Aerial Images using a Deep Learning Approach*, Ecological Informatics, 2019.