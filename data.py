import os
import re
import numpy as np

pattern = re.compile('(.*)"text": "(.*)"(.*)')

def prepare_data(phase):
    filename = './data/raw_data/' + phase + '_data'
    sentence_outname = './data/sentence/' + phase + '_sentence.txt'
    if phase != 'test':
        label_outname = './data/label/' + phase + '_label.npy'
        label_out = []
    file = open(filename, 'r')
    sentence_out = open(sentence_outname, 'w')
    
    while True:
        line = file.readline()
        if not line:
            break
        
        if phase != 'test':
            label_out.append(int(line[11]))
        match = re.match(pattern, line)
        if match:
            words = match[2].split()
            tmp = []
            for word in words:
                if (ord(word[0]) >= 48 and ord(word[0]) <= 57) or (ord(word[0]) >= 65 and ord(word[0]) <= 90) or (ord(word[0]) >= 97 and ord(word[0]) <= 122):
                    tmp.append(word)
            for i in tmp:
                sentence_out.write(i + ' ')
            sentence_out.write('\n')
    
    file.close()
    sentence_out.close()
    if phase != 'test':
        label_out = np.array(label_out)
        np.save(label_outname, label_out)

if __name__ == '__main__':
    
    for phase in ['train', 'test', 'valid']:
        prepare_data(phase)