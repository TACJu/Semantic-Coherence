import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

def build_model():
    
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=2, activation='relu', input_shape=(10, 10, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model

if __name__ == "__main__":

    X_train = np.load('./data/embedding vector/train_vec.npy')
    Y_train = np.load('./data/label/train_label.npy')
    X_val = np.load('./data/embedding vector/valid_vec.npy')
    Y_val = np.load('./data/label/valid_label.npy')

    X_train = np.reshape(X_train, (80000, 10, 10))
    X_val = np.reshape(X_val, (10000, 10, 10))
    X_train = np.expand_dims(X_train, axis=3)
    X_val = np.expand_dims(X_val, axis=3)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=32, epochs=30, verbose=1, validation_data=(X_val, Y_val))


