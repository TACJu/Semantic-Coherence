import numpy as np
from gensim.models import word2vec


def train():
    sentence = word2vec.LineSentence('./data/sentence/train_sentence.txt')
    model = word2vec.Word2Vec(sentences = sentence, min_count = 3)
    model.save('./model/word2vec.model')

def inference(phase):
    model = word2vec.Word2Vec.load('./model/word2vec.model')
    filename = './data/sentence/' + phase + '_sentence.txt'
    file = open(filename, 'r')
    vec = []
    
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
        vec.append(tmp / count)

    vec = np.array(vec)    
    file.close()
    savename = './data/embedding vector/' + phase + '_vec.npy'
    np.save(savename, vec)

if __name__ == "__main__":
    # train()
    for phase in ['train', 'test', 'valid']:
        inference(phase)