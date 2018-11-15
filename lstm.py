import numpy as np
import os
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Embedding, Dense, Flatten

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
        plt.savefig('./lstm_train.png')

def build_model(embedding_matrix):
    
    model = Sequential()
    
    model.add(Embedding(input_dim=86678, output_dim=100, weights=[embedding_matrix], input_length=1000, mask_zero=False, trainable=False))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()

    return model

def test_model(embedding_matrix):
    
    embedding_layer = Embedding(86678, 100, weights=[embedding_matrix], input_length=1000, trainable=False)
    sequence_input = Input(shape=(1000,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    return model

if __name__ == "__main__":

    history = LossHistory()
    X_train = np.load('./data/word vector/train_index_fix.npy')
    Y_train = np.load('./data/label/train_label.npy')
    X_val = np.load('./data/word vector/valid_index_fix.npy')
    Y_val = np.load('./data/label/valid_label.npy')
    embedding_matrix = np.load('./data/word vector/word_vector_fix.npy')
    
    model = build_model(embedding_matrix)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    tensorboard = TensorBoard('./lstm_log/')
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), callbacks=[tensorboard, history], epochs=10, batch_size=100)
    history.loss_plot('epoch')
        
    