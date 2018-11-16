import numpy as np
import os

import keras
from keras import backend as K
from keras import initializers,regularizers,constraints
from keras.engine.topology import Layer
from keras.models import load_model
from keras.utils import CustomObjectScope

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def get_result():
    
    #global val_data
    global test_data
    global model_list

    batch_size = 100
    
    out_label = np.zeros((len(model_list), len(test_data), 1))

    for i, model_name in enumerate(model_list):
        print(i)
        with CustomObjectScope({'Attention': Attention()}):
            model = load_model(model_name)
        out_label[i] = np.round(model.predict(test_data, batch_size))
        

    np.save('./result/test_result.npy', out_label)
    print('save done')

def test():
    
    #global val_data, val_label
    global test_data
    global model_list

    out_label = np.load('./result/test_result.npy')
    
    '''
    for i in range(len(model_list)):
        tp, fp, tn, fn = 0, 0, 0, 0
        for j in range(len(val_data)):
            if val_label[j] == 1:
                if out_label[i][j][0] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if out_label[i][j][0] == 1:
                    fp += 1
                else:
                    tn += 1
        acc = (tp + tn) / (tp + fp + tn + fn)
        print(tp, fp, tn, fn, acc)
    '''

    fuse_predict = np.zeros((1, len(test_data), 1), dtype=np.int)
    for i in range(len(test_data)):
        fuse_predict[0][i][0] = np.argmax(np.bincount([out_label[j][i][0] for j in range(len(out_label))]))

    '''
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(val_data)):
        if val_label[i] == 1:
                if fuse_predict[0][i][0] == 1:
                    tp += 1
                else:
                    fn += 1
        else:
            if fuse_predict[0][i][0] == 1:
                fp += 1
            else:
                tn += 1
    acc = (tp + tn) / (tp + fp + tn + fn)
    print(tp, fp, tn, fn, acc)
    '''

    count0 = 0
    count1 = 0
    file = open('./result/test_predict.txt', 'w')
    
    for i in range(len(test_data)):
        if fuse_predict[0][i][0] == 0:
            count0 += 1
        else:
            count1 += 1
        file.write(str(fuse_predict[0][i][0]) + '\n')
    
    print(count0, count1)
    file.close()
        
if __name__ == "__main__":
    
    #val_data = np.load('./data/word vector/valid_index_fix.npy')
    #val_label = np.load('./data/label/valid_label.npy')
    test_data = np.load('./data/word vector/test_index_fix.npy')
    #val_data = np.expand_dims(val_data, axis=1)
    test_data = np.expand_dims(test_data, axis=1)

    model_list = ['./model/adam_model/model_03-0.67.hdf5', './model/adam_model/model_04-0.68.hdf5', './model/adam_model/model_05-0.67.hdf5',
            './model/adam_model/model_06-0.67.hdf5', './model/adam_model/model_07-0.67.hdf5']

    get_result()
    test()

            



