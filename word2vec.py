import numpy as np
from gensim.models import word2vec


def train():
    sentence = word2vec.LineSentence('./data/sentence/train_sentence.txt')
    model = word2vec.Word2Vec(sentences = sentence, min_count = 3, window=10)
    model.save('./model/word2vec_new.model')

def inference_(phase):
    model = word2vec.Word2Vec.load('./model/word2vec_new.model')
    filename = './data/sentence/' + phase + '_sentence.txt'
    file = open(filename, 'r')
    array = []
    
    while True:
        line = file.readline()
        count = 0
        if not line:
            break

        vec = np.zeros(100)
        words = line.split()
        for word in words:
            if word in model.wv:
                count += 1
                vec += model.wv[word]
        vec /= count
        array.append(vec)
    
    array = np.array(array)
    print(array.shape)    
    file.close()
    savename = './data/embedding vector/' + phase + '_new_vec.npy'
    np.save(savename, array)

def test():
    
    model = word2vec.Word2Vec.load('./model/word2vec_new.model')
    
    a = []
    word_vector = []
    word_vector.append(np.zeros(100))
    a.append('')
    for word in model.wv.vocab.keys():
        a.append(word)
        word_vector.append(model.wv[word])
    
    word_vector = np.array(word_vector)
    np.save('./data/word vector/word_vector_new.npy', word_vector)
    

    for phase in ['train', 'test', 'valid']:
        filename = './data/sentence/' + phase + '_sentence.txt'
        file = open(filename, 'r')
        index = []
        linecount = 0

        while True:
            line = file.readline()
            linecount += 1
            print(linecount)
            tmp = np.zeros(1000)
            wordcount = 0
            if not line:
                break

            words = line.split()
            for word in words:
                if word in model.wv.vocab.keys():
                    tmp[wordcount] = a.index(word)
                    wordcount += 1
            index.append(tmp)

        index = np.array(index)
        print(index.shape)
        savename = './data/word vector/' + phase + '_index_new.npy'
        np.save(savename, index)

if __name__ == "__main__":
    
    train()
    for phase in ['train', 'test', 'valid']:
        inference_(phase)
    test()
    
    train_label = np.load('./data/label/train_label.npy')
    val_label = np.load('./data/label/valid_label.npy')
    
    train_len = len(train_label)
    val_len = len(val_label)
