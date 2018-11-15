import numpy as np
import imp
from sklearn.svm import *

if __name__ == "__main__":
    
    X_train = np.load('./data/embedding vector/train_new_vec.npy')
    Y_train = np.load('./data/label/train_label.npy')
    X_val = np.load('./data/embedding vector/valid_new_vec.npy')
    Y_val = np.load('./data/label/valid_label.npy')
    print('load done')

    svm = NuSVC(max_iter=1000, nu=0.50)
    svm.fit(X_train, Y_train)
    print(svm.score(X_val, Y_val))