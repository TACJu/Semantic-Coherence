import numpy as np
from gensim.models import word2vec


def train():
    sentence = word2vec.LineSentence('./data/train_sentence.txt')
    model = word2vec.Word2Vec(sentences = sentence, min_count = 3)
    model.save('./model/word2vec.model')

def inference():
    model = word2vec.Word2Vec.load('./model/word2vec.model')
    file = open('./data/train_sentence.txt', 'r')
    vec_sum = []
    vec_mean = []
    
    while True:
        line = file.readline()
        tmp = np.array([0 for i in range(100)], dtype = np.float)
        count = 0
        if not line:
            break

        words = line.split()
        for word in words:
            if word in model.wv:
                count += 1
                tmp += model.wv[word]
        vec_sum.append(tmp)
        vec_mean.append(tmp / count)

    vec_sum = np.array(vec_sum)
    vec_mean = np.array(vec_mean)    
    file.close()
    np.save('./data/vec_sum.npy', vec_sum)
    np.save('./data/vec_mean.npy', vec_mean)

if __name__ == "__main__":
    # train()
    inference()