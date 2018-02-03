from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

def original(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', input_shape=(input_dim, input_dim, 3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    # model.add(Dense(1, activation='linear'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def gray(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    # model.add(Dense(1, activation='linear'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def gray2(input_dim):
    model = Sequential()
    model.add(Convolution2D(16, (9, 9), padding='same', input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def more_conv(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(8, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def more_conv_multiple(input_dim, n_parameters):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(8, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))

    model.add(Dense(n_parameters, activation='linear'))

    model.compile(loss='mse', optimizer='adam')

    return model

def more_conv2(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(8, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(16, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def more_conv3(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(8, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(16, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def more_conv4(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(8, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(16, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(32, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model
