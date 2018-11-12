import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Embedding, Masking, Dense, Flatten,Conv1D, MaxPooling1D 
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

def build_model(word_vector):
    
    model = Sequential()

    model.add(Embedding(input_dim=86678, output_dim=100, weights=[word_vector], input_length=1000, mask_zero=True, trainable=False))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    #model.summary()
    return model

def test_model(word_vector):
    embedding_layer = Embedding(86678, 100, weights=[word_vector], input_length=1000, trainable=True)
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

    X_train = np.load('./data/word vector/train_index_fix.npy')
    Y_train = np.load('./data/label/train_label.npy')
    X_val = np.load('./data/word vector/valid_index_fix.npy')
    Y_val = np.load('./data/label/valid_label.npy')
    word_vector = np.load('./data/word vector/word_vector_fix.npy')
    print(len(word_vector))


    #X_train = pad_sequences(sequences=X_train, maxlen=1000, value=0, padding='post', dtype='float32')
    #X_val = pad_sequences(sequences=X_val, maxlen=1000, value=0, padding='post', dtype='float32')
    #X_train = np.expand_dims(X_train, axis=3)
    #X_val = np.expand_dims(X_val, axis=3)
    #print(X_train[0])
    '''
    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999)
    model = build_model(word_vector)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=16, epochs=30, verbose=1, validation_data=(X_val, Y_val))
    '''
    model = test_model(word_vector)
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=2, batch_size=128)
    model.save('./model/model.h5')
    
    model = load_model('./model/model.h5')
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=2, batch_size=64)
    model.save('./model/model.h5')

        
    