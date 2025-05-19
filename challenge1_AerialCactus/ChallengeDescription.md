# Challenge 1:: Aerial Imagery
## Overview
To assess the impact of climate change on Earth's flora and fauna, it is vital to quantify how human activities such as logging, mining, and agriculture are impacting our protected natural areas. Researchers in Mexico have created the VIGIA project, which aims to build a system for autonomous surveillance of protected areas. A first step in such an effort is the ability to recognize the vegetation inside the protected areas. In this competition, you are tasked with creation of an algorithm that can identify a specific type of cactus in aerial imagery.

This is a competition that has been inspired by an existing competition on Kaggle.


## Dataset Description
Download dataset: https://www.kaggle.com/datasets/michiard/aerial-cactus
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a columnar cactus (Neobuxbaumia tetetzo). Images have been from the original dataset to make them uniform in size. The file name of an image corresponds to its id.

You must create a classifier capable of predicting whether an images contains a cactus.
### Files
- train/ - the training set images
- test/ - the test set images (you must predict the labels of these)
- train.csv - the training set labels, indicates whether the image has a cactus (has_cactus = 1)

The sample_submission.csv file is currently not used: it is a sample submission file in the correct format, that is used in automatic evaluation platforms such as Kaggle. You can discard this file.

Original approaches: do not hesitate to go crazy! What about using an existing "Visual Question Answering" LLM model to "synthetically label" the files in the test directory? Why not! Try!!

## Metrics
For this challenge, you are required to define an appropriate metric. You face a binary classification problem, but at this stage you do not know if your data is balanced. Throughout your work, especially the data analysis part, you will first need to characterize your data, and then come up with a valid performance metric to assess the quality of your model.

