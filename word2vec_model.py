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
    array = []
    
    while True:
        line = file.readline()
        tmp = np.array([0 for i in range(100)], dtype = np.float)
        count = 0
        if not line:
            break

        vec = []
        words = line.split()
        for word in words:
            if word in model.wv:
                count += 1
                vec.append(np.sum(model.wv[word]))

        vec = np.array(vec)
        array.append(vec)

    array = np.array(array)
    print(array.shape)    
    file.close()
    savename = './data/embedding vector/' + phase + '_array.npy'
    np.save(savename, array)

def test():
    
    model = word2vec.Word2Vec.load('./model/word2vec.model')
    
    a = []
    #word_vector = []
    #word_vector.append(np.zeros(100))
    a.append('')
    for word in model.wv.vocab.keys():
        a.append(word)
        #word_vector.append(model.wv[word])
    
    #word_vector = np.array(word_vector)
    #np.save('./data/word vector/word_vector_fix.npy', word_vector)
    

    for phase in ['train', 'test', 'valid']:
        filename = './data/sentence/' + phase + '_sentence.txt'
        file = open(filename, 'r')
        index = []
        linecount = 0

        while True:
            line = file.readline()
            linecount += 1
            print(linecount)
            tmp = []
            if not line:
                break

            words = line.split()
            for word in words:
                if word in model.wv.vocab.keys():
                    tmp.append(a.index(word))
            index.append(tmp)

        index = np.array(index)
        print(index.shape)
        savename = './data/word vector/' + phase + '_index_no_zero.npy'
        np.save(savename, index)



if __name__ == "__main__":
    # train()
    #for phase in ['train', 'test', 'valid']:
    #    inference(phase)
    test()