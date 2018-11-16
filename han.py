import numpy as np
import os
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras import initializers,regularizers,constraints
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import Dense, Input, Embedding, GRU, Bidirectional, TimeDistributed
from keras.callbacks import TensorBoard, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
plt.switch_backend('agg')

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('./pictures/han_train.png')

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

def build_model(embedding_matrix):
    
    embedding_layer = Embedding(86678, 100, weights=[embedding_matrix], input_length=1000, trainable=True, mask_zero=True)

    word_input = Input(shape=(1000,), dtype='int32')
    embedded_sequences = embedding_layer(word_input)
    lstm_word = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    #word_dense = TimeDistributed(Dense(200))(lstm_word)
    attn_word = Attention()(lstm_word)
    sentenceEncoder = Model(sentence_input, attn_word)

    sentence_input = Input(shape=(1, 1000), dtype='int32')
    sentence_encoder = TimeDistributed(sentenceEncoder)(sentence_input)
    lstm_sentence = Bidirectional(GRU(100, return_sequences=True))(sentence_encoder)
    #sentence_dense = TimeDistributed(Dense(200))(lstm_word)
    attn_sentence = Attention()(lstm_sentence)
    pred = Dense(1, activation='sigmoid')(attn_sentence)
    model = Model(sentence_input, pred)

    model.summary()

if __name__ == "__main__":

    history = LossHistory()

    X_train = np.load('./data/word vector/train_index_fix.npy')
    Y_train = np.load('./data/label/train_label.npy')
    X_val = np.load('./data/word vector/valid_index_fix.npy')
    Y_val = np.load('./data/label/valid_label.npy')
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    embedding_matrix = np.load('./data/word vector/word_vector_fix.npy')

    model = build_model(embedding_matrix)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    filepath='./model/adam_model/model_{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    tensorboard = TensorBoard('./adam_log/', write_graph=True)

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), callbacks=[checkpoint, tensorboard, history], epochs=10, batch_size=100)
    history.loss_plot('epoch')
