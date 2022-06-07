# Fake News Detector
# LSTM WITH PRETRAINED GLOVE EMBEDDING
# CNN WITH WORD EMBEDDING
# XGBOOST WITH TF-IDF

SUMMARY

Dataset: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

Download dataset Fake and Real news from the source and upload to colab environment

Dataset downsampled to 0.3 for performances purposes

Please run with GPU

These training times below measured with GPU accelaretor

If we don't use GPU the training times increase at least 10 times

Used Google Colab

LSTM MODEL

Training time : 3min

F1 Score : 0.9313

Accuracy score : 0.9301

Only one LSTM layer used with 128 filters

Glove embeddings with 50 dimension

CNN MODEL

Training time : 45 seconds

F1 score : 0.9839

Accuracy Score : 0.9840

Only one 1D Convolution layer used with 128 filters

Embedding layer trained with 30 dimension

XGBOOST MODEL

Training time : 1min 27 seconds

F1 Score : 0.9944

Accuracy Score : 0.9946

We use LSTMs for sequential data processing similar to RNNs. The advantage of LSTM that it introduces gates to overcome RNN's Vanishing Gradient Problem. So it can hold the importance of a word in a sequence, without causing the vanishing of the particular weight. However, both RNN and LSTM need to be computed sequentially so it can't be parallelized to improve calculation speed. Also it requires a lot of memory therefore the training time is longer than other models.

CNNs are usually used in image processing. Fundamentally what a CNN layer does, it gets the most important features in a given dataset, therefore a 1D CNN can be used in a sequential text data to get the most important words. For classification it performs well than even more complex deep learning models, like RNN, LSTM or GRU. It can also be parallelized since it doesn't need to wait for previous information like in LSTM. That makes CNN more efficient. In our results we can see that it completed training much faster than LSTM also f1 and accuracy score are much higher than LSTM model.

XGBoost ml lib is very famous at the moment. It is a descision tree based ensemble algorithm that has high performance on any supervised ml tasks. It generally outperforms the deep learning models. In our results, compared to other two deep learning models, XGBoost performed the best with the highest f1 and accuracy score, respectively 0.9944, 0.9946. It took longer than CNN model to train however we didn't do any hyperparameter tuning and it is way less complex to create a XGBoost model, compared to deep learning models.

As a result, for the given task I would prefer to use XGBoost model since it is less complex to build and it performs much better compared to other two deep learning models.
