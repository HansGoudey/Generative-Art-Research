from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import errno
import Parameter_Support

IMAGE_DIR = "E:\\Generative-Art-Research\\Images\\Roundness"
N_EPOCHS = 1
BATCH_SIZE = 400

RUN_ID = 11



# MAKE A NEW DIRECTORY FOR THE TRAINING RESULTS
NEW_DIR_NAME = 'results_' + str(N_EPOCHS) + '_epochs_' + 'id_' + str(RUN_ID)
try:
    os.makedirs(NEW_DIR_NAME)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise



# CREATE THE MODEL
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
# model.add(LeakyReLU(alpha=0.3))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1, activation='linear'))


model.compile(loss='mse', optimizer='adam')
print(model.summary())



# GATHER THE DATA AND LABELS FROM A DIRECTORY
image_names = os.listdir(IMAGE_DIR)
np.random.shuffle(image_names)
n_names = len(image_names)

x_train = np.array([Parameter_Support.get_image(os.path.join(IMAGE_DIR, name)) for name in image_names[0:int(n_names * 0.8)]]).astype(np.float32)
y_train = np.array([float(name[5:7]) for name in image_names[0:int(n_names * 0.8)]]).reshape(int(n_names * 0.8), 1).astype(np.float32)

x_test = np.array([Parameter_Support.get_image(os.path.join(IMAGE_DIR, name)) for name in image_names[int(n_names * 0.8):n_names]]).astype(np.float32)
y_test = np.array([float(name[5:7]) for name in image_names[int(n_names * 0.8):n_names]]).astype(np.float32)
y_test = y_test.reshape(y_test.shape[0], 1)

print('x test shape: ', x_test.shape)



# TRAIN ON THE TRAINING PORTION
model.fit(x_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE)

train_predictions = model.predict(x_train, batch_size=(BATCH_SIZE//40))
train_predictions_and_y = np.hstack((train_predictions, y_train))

np.savetxt(os.path.join(NEW_DIR_NAME, 'train_results_' + str(N_EPOCHS) + '_epochs_' + 'id_' + str(RUN_ID) + '.csv'), train_predictions_and_y, delimiter=',')

train_score = model.evaluate(x_train, y_train, batch_size=(BATCH_SIZE//40))
print('Train Score: ', train_score)


# TEST ON THE TEST DATA AND SAVE PREDICTIONS
test_predictions = model.predict(x_test, batch_size=(BATCH_SIZE//40))
test_predictions_and_y = np.hstack((test_predictions, y_test))

np.savetxt(os.path.join(NEW_DIR_NAME, 'test_results_' + str(N_EPOCHS) + '_epochs_' + 'id_' + str(RUN_ID) + '.csv'), test_predictions_and_y, delimiter=',')

test_score = model.evaluate(x_test, y_test, batch_size=(BATCH_SIZE//40))
print('Test Score: ', test_score)



# SAVE MODEL FOR FUTURE USE
model.save(os.path.join(NEW_DIR_NAME, 'model_' + str(N_EPOCHS) + '_epochs_' + 'id_' + str(RUN_ID) + '.h5'))



# PLOT THE TRAIN AND TEST PREDICTIONS VS ACTUAL VALUES
figure, (train_plot, test_plot) = plt.subplots(1, 2, figsize=(24, 6))

Parameter_Support.plot_results(train_plot, train_predictions_and_y, train_score, 'Roundness, Train')
Parameter_Support.plot_results(test_plot, test_predictions_and_y, test_score, 'Roundness, Test')

plt.legend()

figure.savefig(os.path.join(NEW_DIR_NAME, 'results_figure_' + str(N_EPOCHS) + '_epochs_' + 'id_' + str(RUN_ID) + '.png'), dpi=300)
plt.show()













































