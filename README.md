# Semantic-Coherence

This project aims to judge whether a context is coherent in meaning or not. As far as I know, it's quite a challenging task.

The raw data is in the format of text, I first use [data.py](./data.py) to extract useful information and store them in numpy or txt format. After that, gensim package is used to deploy Word2Vec method on the words extracted before and get the embedding vectors which are also stored in npy files.

I'm planning to use KNN, SVM, LSTM to process this problem.

## KNN Result

All code is in [knn.py](./knn.py). I'm going to test different distance function and results will be updated after experiments. I have to say that the performance is beyond my expection.

|k|distance function|1|3|5|7|9|
|---|---|---|---|---|---|---|
|acc|l1 distance|50.43%|50.44%|50.05%|50.50%|49.94%|
|acc|l1_abs distance|51.21%|52.43%|52.53%|53.33%|53.53%|
|acc|l2 distance|52.19%|52.69%|52.48%|52.96%|53.81%|