#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [Sentimental analysis - Challenge 3 AML],
  abstract: [
    - Sentence-level Sentiment Analysis evaluate sentiment from a single sentence.
  ],
  authors: (
    (
      name: "Alex Argese",
      department: [Student],
      organization: [EURECOM],
      location: [Biot, France],
      email: "alex.argese@eurecom.fr",
    ),
    (
      name: "Cristian Degni",
      department: [Student],
      organization: [EURECOM],
      location: [Biot, France],
      email: "cristian.degni@eurecom.fr",
    ),
    (
      name: "Miriam Lamari",
      department: [Student],
      organization: [EURECOM],
      location: [Biot, France],
      email: "miriam.lamari@eurecom.fr",
    ),
    (
      name: "Enrico Sbuttoni",
      department: [Student],
      organization: [EURECOM],
      location: [Biot, France],
      email: "enrico.sbuttoni@eurecom.fr",
    ),
  ),
  index-terms: ("Machine Learning", "Sentimental Analysis", "Natural Language Process", "", ""), //TODO: add more index terms,
)

= Introduction
Sentiment analysis is the process of determining the opinion, judgment, or emotion behind natural language. It can be a can be a very powerful technique since it is widely applied to voice of the customer materials such as reviews and survey responses, online and social media. The most advanced sentiment analysis can identify precise emotions like anger, sarcasm, confidence or frustration. In this challenge, we focus on sentence-level sentiment analysis, which evaluates sentiment from a single sentence. The primary goal is to classify sentences into one of three categories: *positive*, *negative*, or *neutral*. The dataset used for this task consists of *tweets from Figure Eight's Data* for Everyone platform.

= Dataset

The dataset is provided in CSV format, with each row representing a tweet and its associated metadata.
The training set contains *24732 samples*, while the test set contains *2748 samples*. The dataset is available on the *Figure Eight* platform, which provides a wide range of datasets for various machine learning tasks.
Inside training set, four columns are present:
- *textID*: unique identifier for each tweet;
- *text*: the tweet text;
- *selected_text*: the text selected by the annotator as the most relevant part of the tweet for sentiment analysis;
- *sentiment*: the sentiment label assigned to the tweet, which can be one of three classes: *positive*, *negative*, or *neutral*.

The *class distribution* within the training set is *unbalanced*:
- *positive*: 7711 samples
- *negative*: 7003 samples
- *neutral*: 10018 samples


#figure(
  image("img/distribution.png", width: 85%),
  caption: [
    Train samples class distribution.
  ],
)

= Preprocessing
In order to prepare the dataset for training, we performed several preprocessing steps on the text data:
- *removing URLs, mentions and hashtags*: URLs, mentions, and hashtags were removed from the text to focus on the actual content of the tweet;
- *removing punctuation and special characters*: punctuation marks and special characters were removed to simplify the text and reduce noise;
- *conversion of numbers into words*: numbers were converted into their word equivalents to maintain consistency in the text;
- *lowercase*: all text was converted to lowercase to ensure uniformity and avoid case sensitivity issues;
- *removing stop words*: common words that do not contribute to the sentiment analysis were removed to reduce noise and improve model performance. Words of negation such as "not" and "no" were kept, as they can significantly impact the sentiment of a sentence;
- *tokenization*: the text was tokenized into individual words to facilitate further processing and analysis.

It follows the WordCloud visualization of the most frequent words in the training set, depending on the class they belongs to, which highlights the most common terms used in the tweets. This visualization can help identify key themes and topics present in the dataset.

#let vimg(body) = {
    rect(width: 10mm, height: 5mm)[
        #text(body)
    ]
}

#figure(
  grid(
    columns: (auto, auto, auto),
    rows: (auto),
    gutter: 1em,
    [
      #image("img/negative_wordcloud.png", width: 80%),
      #image("img/neutral_wordcloud.png", width: 80%),
      #image("img/positive_wordcloud.png", width: 80%),
    ],
  ),
  caption: [Word clouds for the negative, neutral, and positive classes.]
)




= Models Evaluated

== GRU-Based Sentiment Classifier

A *Gated Recurrent Unit (GRU)*-based model was developed using *TensorFlow/Keras* to classify *tweet sentiment* from *preprocessed text data*. Each input is first transformed into padded token sequences (`max_len = 50`) and passed through an *Embedding layer* (`output_dim=128`). The embedded sequences are then processed by a *GRU layer* with *64 units*, followed by a *Dropout layer* (`rate=0.5`) to mitigate overfitting. A final *Dense layer* with *softmax activation* predicts one of the *three sentiment classes*.

The training process uses *sparse categorical crossentropy* as the loss function and the *Adam optimizer* with a *batch size of 32* for *10 epochs*. Model evaluation on the test set includes computing the *macro F1 score* and generating a *detailed classification report*. Labels are encoded using *LabelEncoder* to match the expected format for training.

#figure(
  caption: [Hyperparameter setup for GRU-based Sentiment Classifier],
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: (left, left, center),
    table.header(
      [Hyperparameter],
      [Ranges / Values],
      [Used],
    ),

    [embedding_dim], [128], [128],
    [gru_units], [32, 64, 128], [64],
    [dropout], [0.3–0.5], [0.5],
    [optimizer], [Adam], [Adam],
    [epochs], [10–30], [10],
    [batch_size], [32, 64], [32],
  ),
) <tab:gru-params>


== RNN-Based Sentiment Classifier

A *Recurrent Neural Network (RNN)*, specifically a *LSTM-based classifier*, was implemented using *TensorFlow/Keras* to perform *sentiment classification* on *preprocessed tweet data*. The architecture begins with a *tokenized and padded input sequence* (max length = *50*), embedded into a *dense vector space* via an *Embedding layer* (`output_dim=128`). This is followed by a *single LSTM layer* (`units=64`) and a *Dropout layer* (`rate=0.5`) to reduce overfitting. The final *Dense output layer* uses a *softmax activation* for multi-class classification across *three sentiment labels*.

The model is trained with *sparse categorical crossentropy* loss and optimized using the *Adam optimizer* over *10 epochs*. Performance is evaluated using *macro-averaged F1 score* and a detailed *classification report*, with results obtained on a held-out *test set*. Input labels are encoded numerically using *sklearn’s LabelEncoder* for compatibility with the loss function.

#figure(
  caption: [Hyperparameter setup for LSTM-based Sentiment Classifier],
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: (left, left, center),
    table.header(
      [Hyperparameter],
      [Ranges / Values],
      [Used],
    ),

    [embedding_dim], [128], [128],
    [lstm_units], [32, 64, 128], [64],
    [dropout], [0.3–0.5], [0.5],
    [optimizer], [Adam], [Adam],
    [epochs], [10–30], [10],
    [batch_size], [32, 64], [32],
  ),
) <tab:rnn-params>


== RNN with Self-Attention Sentiment Classifier

A *Recurrent Neural Network* enhanced with a *Self-Attention mechanism* was constructed using *TensorFlow/Keras* to classify *tweet sentiment*. Each input sentence is tokenized and padded (`max_len = 50`), then embedded via an *Embedding layer* (`output_dim=128`). This is followed by a *GRU layer* (`units=64`) whose output is fed into a *custom Self-Attention layer*. The attention mechanism computes *context-aware weighted representations* of the input sequence to improve the model’s focus on relevant tokens. The result is aggregated and passed to a *Dense softmax layer* for final classification into *three sentiment classes*.

The model is trained using the *Adam optimizer* and *sparse categorical crossentropy* loss function for *10 epochs*, with *batch size 32*. Labels are numerically encoded using *LabelEncoder*. The classifier’s performance is assessed through a *macro-averaged F1 score* and a *classification report* on a held-out test set.

#figure(
  caption: [Hyperparameter setup for RNN with Self-Attention Sentiment Classifier],
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: (left, left, center),
    table.header(
      [Hyperparameter],
      [Ranges / Values],
      [Used],
    ),

    [embedding_dim], [128], [128],
    [gru_units], [32, 64, 128], [64],
    [attention], [Custom Layer], [✓],
    [optimizer], [Adam], [Adam],
    [epochs], [10–30], [10],
    [batch_size], [32, 64], [32],
  ),
) <tab:attn-rnn-params>


== RoBERTa-Based Sentiment Classifier

A *Transformer-based model*, specifically *RoBERTa fine-tuned for sentiment analysis*, was implemented using the *Hugging Face Transformers* library. The model used is `"cardiffnlp/twitter-roberta-base-sentiment"`, pre-trained on social media text. Input sentences are first *tokenized* using the corresponding *AutoTokenizer*, padded and truncated appropriately, and then passed to the *RoBERTa model* which outputs contextualized representations. A classification head maps these to *three sentiment classes*.

The model is trained using the *Trainer API* with *weighted F1 score*, *accuracy*, *precision*, and *recall* as evaluation metrics. Data is managed using the *Hugging Face Datasets* library and split into *70% training*, *15% validation*, and *15% test* sets. Labels are encoded via *LabelEncoder* to match the model’s expected format.

#figure(
  caption: [Hyperparameter setup for RoBERTa Sentiment Classifier],
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: (left, left, center),
    table.header(
      [Hyperparameter],
      [Ranges / Values],
      [Used],
    ),

    [batch_size], [8, 16, 32], [default],
    [epochs], [3–5], [default],
    [max_length], [auto via tokenizer], [✓],
  ),
) <tab:roberta-params>


== TF-IDF + Traditional Classifiers for Sentiment Analysis

A *classic machine learning pipeline* was employed to classify *tweet sentiment* using a *TF-IDF vectorization* of preprocessed text. Input text is tokenized, cleaned, and then transformed into a *TF-IDF matrix* (`max_features=5000`) using *scikit-learn’s TfidfVectorizer*. The resulting sparse matrix represents word importance across documents and serves as input to various *supervised classifiers*.

Multiple models were evaluated, including *Logistic Regression*, *Linear SVM (SVC)*, *Multinomial Naive Bayes*, and *Random Forests*. Each model is trained on *70% of the data*, validated on *15%*, and tested on the remaining *15%*. The primary metric for evaluation is the *macro-averaged F1 score*, complemented by *classification reports* and *confusion matrices*.

#figure(
  caption: [Hyperparameter setup for TF-IDF + Traditional Classifiers],
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: (left, left, center),
    table.header(
      [Component],
      [Details],
      [Used],
    ),

    [vectorizer], [TfidfVectorizer], [✓],
    [max_features], [1000–10000], [5000],
    [classifiers], [LogReg, SVM, RF, NB], [✓],
    [split ratio], [70/15/15], [✓],
    [loss / objective], [varies by model], [✓],
    [evaluation], [F1-macro, ConfMatrix], [✓],
  ),
) <tab:tfidf-params>




= Metric Justification

To evaluate model performance in ranking anomalies, we primarily used the *ROC AUC*, which measures the model's ability to distinguish between *normal* and *anomalous* samples, independently of a decision threshold. This is particularly important in *unsupervised anomaly detection*, where decision boundaries are not known a priori.

While additional classification metrics such as *F1-score*, *precision*, and *recall* were reported after threshold selection, they were not used for *model selection* or hyperparameter tuning.



= Results Summary

== Detailed Results – Fully-Connected Encoder

Prior to implementing the *Variational Autoencoder*, we trained a simple *fully-connected encoder* on the same *normalized tabular Mel spectrograms*. Unlike the *VAE*, this model did not include a *generative decoder*; instead, it focused purely on *compressing input representations*. *Anomaly scores* were computed using the *Mean Squared Error (MSE)* between the *encoded representations* of test samples and the *mean embedding vector* of the training data. Despite its simplicity, this approach achieved a *ROC AUC of 0.7109*, highlighting that even a *basic encoder* can yield *meaningful representations* for *anomaly detection* when combined with a *suitable scoring function*.


== Detailed Results – Fully-Connected VAE

The *fully-connected Variational Autoencoder* (*VAE*) was trained on *normalized tabular representations* of the Mel spectrograms. Despite effective training convergence (with total loss decreasing from over *10,000* to below *1,000*) the final *ROC AUC* was only *0.3975*. This suggests the learned embeddings were *not sufficiently informative* for distinguishing anomalous samples. The poor performance may stem from the VAE's *limited capacity to model temporal and spatial structure* in spectrogram data.

== Detailed Results – Convolutional VAE

The *Convolutional VAE* (*ConvVAE*) was trained directly on *2D Mel spectrograms*, using convolutional layers to better capture *local spatial structure*. A grid search over *latent dimensions* and *β values* identified the optimal setting as `latent_dim=32`, `β=0.01`, yielding a *cross-validated loss* of *33689.0*. Final evaluation on the test set produced a *ROC AUC* of *0.7890*, with an *F1-score* of *0.8575*, *accuracy* of *0.7947*, and *precision* of *0.8662*. These results confirm the model's ability to *accurately reconstruct normal patterns* while effectively distinguishing *anomalous events*.

== Detailed Results – PANNs Embedding + Mahalanobis Distance

The *PANNs-based model* used a *pretrained Cnn14 architecture* to extract *2048-dimensional embeddings* from raw waveforms. An anomaly score was computed using the *Mahalanobis distance* from the mean of normal embeddings. This method achieved the best overall result, with a *ROC AUC* of *0.9311*, significantly outperforming the reconstruction-based approaches. The strong performance confirms the effectiveness of *pretrained audio representations* for capturing *semantic structure* in acoustic scenes.

== Detailed Results – VGGish Embedding + Mahalanobis Distance

The *VGGish model* was used to generate *128-dimensional embeddings* from the audio inputs. *Anomaly detection* was performed by computing the *Mahalanobis distance* between each test embedding and the distribution of embeddings obtained from *normal training data*. This method achieved a *ROC AUC of 0.8513*, confirming the effectiveness of *transfer learning* and *pretrained audio representations* for detecting *anomalous acoustic events*. While not as performant as *PANNs*, the results show that *VGGish embeddings* still capture *relevant structure* in the data, providing a strong balance between *efficiency* and *accuracy*.

== *Metric Selection Rationale*

For both *VGGish* and *PANNs-based models*, we experimented with different *anomaly scoring methods*, including *Euclidean* and *cosine distances*. However, we observed that using the *Mahalanobis distance* consistently yielded significantly better results—improving *ROC AUC scores* by approximately *10 percentage points*. This improvement is likely due to *Mahalanobis distance* accounting for the *covariance structure* of the *embedding space*, making it more sensitive to *deviations from the normal distribution* in *high-dimensional representations*.


#figure(
  caption: [Performance Comparison of Models on the Test Set],
  table(
    columns: (auto, auto),
    align: (left, center),
    table.header[Model][ROC AUC],
    [Fully-Connected Encoder], [0.7109],
    [Fully-Connected VAE], [0.3975],
    [Convolutional VAE], [0.7890],
    [PANNs + Mahalanobis], [0.9311],
    [VGGish + Mahalanobis], [0.8513],
  ),
) <tab:our-results>

= Model Selected


Although the *Convolutional VAE* performed well and achieved high *precision* and *F1-score* on the held-out test set, the *PANNs embedding method* achieved a *substantially higher ROC AUC* of *0.9311*. Given the *unsupervised nature* of the task and the emphasis on *ranking anomaly likelihood*, *ROC AUC* was prioritized as the primary evaluation metric.

/* #figure(
  image("img/mahala.png", width: 90%),
) */

 Therefore, we selected the *PANNs + Mahalanobis* method as the *final model*, due to its superior *generalization*, *semantic awareness*, and *robustness to noise*. This approach is also *computationally efficient*, requiring no retraining and leveraging *powerful pretrained audio features*.


= Inference on Unlabeled Test Set

We applied both the *Conv-VAE* and the *PANNs + Mahalanobis* pipeline to the *unlabeled evaluation set* provided. The goal was to rank test samples by *anomaly score* and identify the most suspicious examples.

== Conv-VAE

The *Convolutional Variational Autoencoder* computes an *anomaly score* based on the *reconstruction error* between the input spectrogram and its reconstruction. A manually selected threshold (derived from the validation ROC curve) is applied to label samples as normal or anomalous.

While the model can flag anomalous instances, it suffers from:
- *sensitivity to audio distortions*,
- potential overfitting to training noise patterns,
- and *limited generalization* to unseen anomalies.

This makes its inference results less reliable compared to embedding-based methods.

== PANNs + Mahalanobis

The *PANNs model* provides *pretrained 2048-dimensional embeddings* for each audio file. By modeling the distribution of normal embeddings and computing the *Mahalanobis distance*, we obtain robust anomaly scores without retraining. This method:
- generalizes well across machine conditions,
- is *insensitive to minor signal variations*,
- and produces *well-calibrated anomaly rankings*.

= Conclusion and Next Steps

In this challenge, we compared three unsupervised approaches for *anomalous sound detection* on the *slider machine* using audio recordings from the MIMII dataset. Our analysis revealed that while *fully-connected VAEs* were limited by their inability to capture spectro-temporal patterns, *Convolutional VAEs* significantly improved anomaly detection by leveraging local structure in Mel spectrograms.

However, the best results were obtained by the *PANNs-based model* with *Mahalanobis distance*, which achieved a *ROC AUC of 0.9311* without requiring retraining. This underscores the value of *pretrained semantic audio representations* for generalization in real-world noisy environments.

As future work, we propose exploring *semi-supervised training* using pseudo-labeling strategies, experimenting with *attention-based models* applying *temporal models* (e.g., LSTMs or Transformers) to capture longer-term dependencies in machine sounds, and evaluating the use of *domain adaptation techniques* to improve robustness across different machine types or noise conditions.


= References

This report is inspired by the DCASE challenge and its application to real-world industrial environments, as described in:

DCASE Challenge Task 2 (2020), *Unsupervised Anomalous Sound Detection for Machine Condition Monitoring*.
The MIMII Dataset: Koizumi et al., *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation*, 2019.
The ToyADMOS Dataset: Purohit et al., *ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection*, 2019.

