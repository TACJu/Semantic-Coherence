# Semantic-Coherence

This project aims to judge whether a context is coherent in meaning or not. As far as I know, it's quite a challenging task.

The raw data is in the format of text, I first use data.py to extract useful information and store them in numpy or txt format. After that, gensim package is used to deploy Word2Vec method on the words extracted before and get the embedding vectors which are also stored in npy files.

I'm planning to use KNN, SVM, LSTM to process this problem.

## KNN Result

All code is in knn.py. I'm going to test different distance function and results will be updated after experiments. I have to say that the performance is beyond my expection.

|k|1|3|5|7|9|
|---|---|---|---|---|---|
|acc|51.21%|52.43%|52.53%|53.33%|53.53%|