from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import errno
import Parameter_Support

IMAGE_DIR = "E:\\Generative-Art-Research\\Images\\SimpleSquareCircle"
IMAGE_TYPES = None  # Add a list of the first three identifier letters at the beginning of the image files to only train on those types
N_EPOCHS = 300
RUN_ID = 18
PREVIOUS_MODEL_TO_LOAD = None  # Add the name of the folder containing the model to load and start the training with

BATCH_SIZE = 400



# MAKE A NEW DIRECTORY FOR THE TRAINING RESULTS
NEW_DIR_NAME = 'Run ' + str(RUN_ID) + ' Results ' + str(N_EPOCHS) + ' Epochs'
try:
    os.makedirs(NEW_DIR_NAME)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise



# CREATE THE MODEL
if PREVIOUS_MODEL_TO_LOAD is None:
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', input_shape=(Parameter_Support.IMAGE_DIM, Parameter_Support.IMAGE_DIM, 3)))
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

    model.add(Dense(1, activation='linear'))


    model.compile(loss='mse', optimizer='adam')
else:
    model = load_model(os.path.join(PREVIOUS_MODEL_TO_LOAD, 'Model.h5'))

with open(os.path.join(NEW_DIR_NAME, 'Model Summary.txt'), 'w+') as summary_file:
    model.summary(print_fn=lambda x: summary_file.write(x + '\n'))



# GATHER THE DATA AND LABELS FROM THE IMAGE DIRECTORY
print('Loading Data')
image_names = os.listdir(IMAGE_DIR)
np.random.shuffle(image_names)
n_names = len(image_names)

if IMAGE_TYPES is None:  # No need to check the type of the image before adding it to data
    x_train = np.array([Parameter_Support.get_image(os.path.join(IMAGE_DIR, name)) for name in image_names[0:int(n_names * 0.8)]]).astype(np.float32)
    y_train = np.array([float(name[5:7]) for name in image_names[0:int(n_names * 0.8)]]).reshape(int(n_names * 0.8), 1).astype(np.float32)

    x_test = np.array([Parameter_Support.get_image(os.path.join(IMAGE_DIR, name)) for name in image_names[int(n_names * 0.8):n_names]]).astype(np.float32)
    y_test = np.array([float(name[5:7]) for name in image_names[int(n_names * 0.8):n_names]]).astype(np.float32)
else:
    print('Only training on types', IMAGE_TYPES)
    x_train = np.array([Parameter_Support.get_image(os.path.join(IMAGE_DIR, name)) for name in image_names[0:int(n_names * 0.8)] if any(kind in name for kind in IMAGE_TYPES)]).astype(np.float32)
    y_train = np.array([float(name[5:7]) for name in image_names[0:int(n_names * 0.8)] if any(kind in name for kind in IMAGE_TYPES)]).reshape(int(n_names * 0.8), 1).astype(np.float32)

    x_test = np.array([Parameter_Support.get_image(os.path.join(IMAGE_DIR, name)) for name in image_names[int(n_names * 0.8):n_names] if any(kind in name for kind in IMAGE_TYPES)]).astype(np.float32)
    y_test = np.array([float(name[5:7]) for name in image_names[int(n_names * 0.8):n_names] if any(kind in name for kind in IMAGE_TYPES)]).astype(np.float32)
y_test = y_test.reshape(y_test.shape[0], 1)



# TRAIN ON THE TRAINING PORTION
history = model.fit(x_train, y_train, validation_split=0.0, epochs=N_EPOCHS, batch_size=BATCH_SIZE)

train_predictions = model.predict(x_train, batch_size=1)
train_predictions_and_y = np.hstack((train_predictions, y_train))

np.savetxt(os.path.join(NEW_DIR_NAME, 'Train Results.csv'), train_predictions_and_y, delimiter=',')

train_score = model.evaluate(x_train, y_train, batch_size=100)
print('Train Score: ', train_score)



# SAVE AND PLOT LOSS HISTORY THROUGH TRAINING EPOCHS
train_loss_history = history.history['loss']
train_loss_history = np.array(train_loss_history).reshape((len(train_loss_history), 1))
figure, plot = plt.subplots(1, 1, figsize=(8, 6))
plot.set_title('Model Loss')
plot.set_xlabel('Epoch')
plot.set_ylabel('Loss')

plot.plot(train_loss_history)

if 'val_loss' in history.history:
    validation_loss_history = history.history['val_loss']
    validation_loss_history = np.array(validation_loss_history).reshape((len(validation_loss_history), 1))
    loss_history = np.hstack((train_loss_history, validation_loss_history))

    plot.plot(validation_loss_history)
    plot.legend(['Train', 'Validation'])
else:
    loss_history = train_loss_history

np.savetxt(os.path.join(NEW_DIR_NAME, 'Loss History.csv'), loss_history, delimiter=',')

figure.savefig(os.path.join(NEW_DIR_NAME, 'Loss History.png'), dpi=300)



# TEST ON THE TEST DATA AND SAVE PREDICTIONS
test_predictions = model.predict(x_test, batch_size=100)
test_predictions_and_y = np.hstack((test_predictions, y_test))

np.savetxt(os.path.join(NEW_DIR_NAME, 'Test Results.csv'), test_predictions_and_y, delimiter=',')

test_score = model.evaluate(x_test, y_test, batch_size=100)
print('Test Score: ', test_score)



# SAVE MODEL FOR FUTURE USE
model.save(os.path.join(NEW_DIR_NAME, 'Model.h5'))



# PLOT THE TRAIN AND TEST PREDICTIONS VS ACTUAL VALUES
figure, (train_plot, test_plot) = plt.subplots(1, 2, figsize=(24, 6))

Parameter_Support.plot_results(train_plot, train_predictions_and_y, train_score, 'Roundness Train')
Parameter_Support.plot_results(test_plot, test_predictions_and_y, test_score, 'Roundness Test')

plt.legend()

figure.savefig(os.path.join(NEW_DIR_NAME, 'Results.png'), dpi=300)
plt.show()













































