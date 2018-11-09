import os
import re

pattern = re.compile('(.*)"text": "(.*)"(.*)')

def prepare_data(phase):
    filename = './data/' + phase + '_data'
    sentence_outname = './data/' + phase + '_sentence.txt'
    label_outname = './data/' + phase + '_label.txt'
    file = open(filename, 'r')
    sentence_out = open(sentence_outname, 'w')
    label_out = open(label_outname, 'w')
    
    while True:
        line = file.readline()
        if not line:
            break
        
        label_out.write(line[11] + '\n')
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
    label_out.close()

def extract():
    stringtoint = {}
    inttostring = {}
    count = 0
    outfile = open('./data/word.txt', 'w')
    
    for phase in ['train', 'test', 'valid']:
        filename = './data/' + phase + '_sentence.txt'
        file = open(filename, 'r')
        
        while True:
            line = file.readline()
            if not line:
                break

            words = line.split()
            for word in words:
                if word not in stringtoint.keys():
                    stringtoint[word] = count
                    inttostring[count] = word
                    count += 1
            
        file.close()
    
    print(len(stringtoint))
    for i in stringtoint:
        outfile.write(i + '\n')
    outfile.close()
    


if __name__ == '__main__':
    '''
    for phase in ['train', 'test', 'valid']:
        prepare_data(phase)
    '''

    extract()