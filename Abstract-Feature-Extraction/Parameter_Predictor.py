from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np
import os
import pickle

IMAGE_DIM = 200
IMAGE_DIR = "C:\\Users\\Hans Goudey\\Documents\\CS500 ML\\Images\\Roundness"
N_EPOCHS = 300
BATCH_SIZE = 50



# CREATE THE MODEL
model = Sequential()
model.add(Convolution2D(32, (9, 9), activation='relu', input_shape=(IMAGE_DIM, IMAGE_DIM, 3)))
model.add(MaxPooling2D())
model.add(Convolution2D(64, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')



# GATHER THE DATA AND LABELS FROM A DIRECTORY
def get_image(path):
    with Image.open(path) as image:
        image = image.resize((IMAGE_DIM, IMAGE_DIM))
        image = np.array(image) / 255
        return image.astype('float32')

image_names = os.listdir(IMAGE_DIR)
np.random.shuffle(image_names)
n_names = len(image_names)

x_train = np.array([get_image(os.path.join(IMAGE_DIR, name)) for name in image_names[0:int(n_names * 0.8)]]).astype(np.float32)
y_train = np.array([float(name[5:7]) for name in image_names[0:int(n_names * 0.8)]]).astype(np.float32)

# print('shape of x_train: ' + str(x_train.shape))

x_test = np.array([get_image(os.path.join(IMAGE_DIR, name)) for name in image_names[int(n_names * 0.8):n_names]]).astype(np.float32)
y_test = np.array([float(name[5:7]) for name in image_names[int(n_names * 0.8):n_names]]).reshape(int(n_names - n_names * 0.8), 1).astype(np.float32)
# print("shape of y_test", y_test.shape)




# TRAIN ON THE TRAINING PORTION
model.fit(x_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE)



# TEST ON THE TEST DATA AND SAVE PREDICTIONS
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print('Score: ', score)

predictions = model.predict(x_test, batch_size=BATCH_SIZE)
# print("shape of predictions", predictions.shape)

both = np.hstack((predictions, y_test))
print("Sample of predictions vs actual values \n", both)

with open('predictions_300_epochs.pickle', 'wb') as handle:
    pickle.dump(both, handle, protocol=pickle.HIGHEST_PROTOCOL)

np.savetxt("both_300_epochs.csv", both, delimiter=",")



# SAVE MODEL FOR FUTURE USE
model.save('model_300_epochs.h5')
