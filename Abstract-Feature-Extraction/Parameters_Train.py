import errno
import os
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import Parameter_Models
import Parameter_Support

IMAGE_DIR = "E:\\Generative-Art-Research\\Images\\Messy Round Rects"
IMAGE_TYPES = ['mes', 'rdr', 'rns', 'rsc', 'img', 'rmr']  # Add a list of the first three identifier letters at the beginning of the image files to only train on those types

PARAMETER_MAP = {'f':'r'}  # Dictionary mapping of equivalent parameters. {'f':'r'} means 'f' should be equivalent to 'r' for training

N_EPOCHS = 1
RUN_ID = 302

MODEL_TO_LOAD = "Run 301 Results 200 Epochs"  # Add the name of the folder containing the model to load and start the training with  'Run 18 Results 300 Epochs'  #
LOAD_PREVIOUS_MODEL = True

BATCH_SIZE = 300



########################################################################################################################
# MAKE A NEW DIRECTORY FOR THE TRAINING RESULTS
########################################################################################################################
NEW_DIR_NAME = 'Run ' + str(RUN_ID) + ' Results ' + str(N_EPOCHS) + ' Epochs'
try:
    os.makedirs(NEW_DIR_NAME)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise



########################################################################################################################
# GATHER THE DATA AND LABELS FROM THE IMAGE DIRECTORY
########################################################################################################################
print('Loading Data')
image_names = os.listdir(IMAGE_DIR)
np.random.shuffle(image_names)

# Remove the images types that aren't wanted for training
image_names = [name for name in image_names if any(kind in name for kind in IMAGE_TYPES)]
n_names = len(image_names)
print('n_images', n_names)

x = np.array([Parameter_Support.get_image(os.path.join(IMAGE_DIR, name)) for name in image_names])
x = x.reshape((n_names, Parameter_Support.IMAGE_DIM, Parameter_Support.IMAGE_DIM, 1)).astype(np.float32)

n_parameters = image_names[0].count('-') - 1
y = np.ones(shape=(n_names, n_parameters), dtype=np.float32) * 50  # Initialize the array to the middle value in case some file names are missing data

# Map the parameters in the parameter map dictionary
temp_image_names = list(image_names)
for name in temp_image_names:
    for letter in PARAMETER_MAP:
        if '-' + letter in name:
            # print('name before replace', name)
            # print('letter', letter, 'is in name')
            name = name.replace('-' + letter, '-' + PARAMETER_MAP[letter])
            # print('name after replace', name)

parameter_indexes = [m.start() + 1 for m in re.finditer('-', temp_image_names[0])][0:-1]  # Don't use the last -, because it's before the last number, not a parameter
parameter_letters = [temp_image_names[0][i] for i in parameter_indexes]
print('parameter letters', parameter_letters)

# Build y
for i_name in range(len(temp_image_names)):
    for i_letter in range(len(parameter_letters)):
        letter = parameter_letters[i_letter]
        index_in_name = temp_image_names[i_name].find('-' + letter)
        if index_in_name == -1:
            # print(image_names[i_name], 'is missing the', letter, 'parameter')
            continue

        value = temp_image_names[i_name][index_in_name + 2:index_in_name + 4]
        y[i_name, i_letter] = value

test_split = int(x.shape[0] * 0.8)

del temp_image_names

x_train, x_test = np.array_split(x, [test_split])
y_train, y_test = np.array_split(y, [test_split])



########################################################################################################################
# CREATE THE MODEL AND SAVE A DESCRIPTION OF THE TRAINING RUN
########################################################################################################################
print('Creating Model')
if LOAD_PREVIOUS_MODEL:
    model = load_model(os.path.join(MODEL_TO_LOAD, 'Model.h5'))
else:
    model = Parameter_Models.more_conv_multiple(Parameter_Support.IMAGE_DIM, n_parameters)

with open(os.path.join(NEW_DIR_NAME, 'Model Summary.txt'), 'w+') as summary_file:
    summary_file.write('\nParameters: ')
    for letter in parameter_letters:
        summary_file.write(letter + ', ')
    summary_file.write('Trained from ' + IMAGE_DIR + '\n')
    if LOAD_PREVIOUS_MODEL:
        summary_file.write('Loaded from previously trained model in ' + MODEL_TO_LOAD + '\n')
    summary_file.write('Valid image types: ')
    for kind in IMAGE_TYPES:
        summary_file.write(kind + ', ')

    summary_file.write('\n\n')

    model.summary(print_fn=lambda x: summary_file.write(x + '\n'))




########################################################################################################################
# SAVE MODEL AND OUTPUT PARAMETERS FOR FUTURE USE
########################################################################################################################
model.save(os.path.join(NEW_DIR_NAME, 'Model.h5'))
with open(os.path.join(NEW_DIR_NAME, 'Parameters.pickle'), 'wb') as parameter_file:
    pickle.dump(parameter_letters, parameter_file)




########################################################################################################################
# TRAIN ON THE TRAINING PORTION OF DATA
########################################################################################################################
history = model.fit(x_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE)

train_predictions = model.predict(x_train, batch_size=BATCH_SIZE)
train_predictions_and_y = np.hstack((train_predictions, y_train))
print('train predictions and y shape', train_predictions_and_y.shape)

np.savetxt(os.path.join(NEW_DIR_NAME, 'Train Results.csv'), train_predictions_and_y, delimiter=',', header=(','.join(parameter_letters) + ',') * 2)

train_score = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)
print('Train Score: ', train_score)


########################################################################################################################
# SAVE AND PLOT LOSS HISTORY THROUGH TRAINING EPOCHS
########################################################################################################################
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

np.savetxt(os.path.join(NEW_DIR_NAME, 'Loss History.csv'), loss_history, delimiter=',', header=(','.join(parameter_letters) + ',') * 2)

figure.canvas.set_window_title('Loss History')
figure.savefig(os.path.join(NEW_DIR_NAME, 'Loss History.png'), dpi=300)



########################################################################################################################
# TEST ON THE TEST DATA AND SAVE PREDICTIONS
########################################################################################################################
test_predictions = model.predict(x_test, batch_size=BATCH_SIZE)
test_predictions_and_y = np.hstack((test_predictions, y_test))

np.savetxt(os.path.join(NEW_DIR_NAME, 'Test Results.csv'), test_predictions_and_y, delimiter=',', header=(','.join(parameter_letters) + ',') * 2)

test_score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print('Test Score: ', test_score)




########################################################################################################################
# PLOT THE TRAIN PREDICTIONS VS ACTUAL VALUES
########################################################################################################################
train_figure, subplots = plt.subplots(1, n_parameters, figsize=(12, 6))  # Create a subplot for each parameter

for i in range(n_parameters):
    n_train = train_predictions_and_y.shape[0]
    parameter_predictions_and_y = np.hstack((train_predictions_and_y[:, i].reshape(n_train, 1), train_predictions_and_y[:, n_parameters + i].reshape(n_train, 1)))
    parameter_predictions_and_y = parameter_predictions_and_y.reshape(n_train, 2)
    Parameter_Support.plot_results(subplots[i], parameter_predictions_and_y, train_score, parameter_letters[i])

train_figure.canvas.set_window_title('Train Results')
train_figure.legend()

train_figure.savefig(os.path.join(NEW_DIR_NAME, 'Train Results.png'), dpi=300)




########################################################################################################################
# PLOT THE TEST PREDICTIONS VS ACTUAL RESULTS
########################################################################################################################
test_figure, subplots = plt.subplots(1, n_parameters, figsize=(12, 6))  # Create a subplot for each parameter

for i in range(n_parameters):
    n_test = test_predictions_and_y.shape[0]
    parameter_predictions_and_y = np.hstack((test_predictions_and_y[:, i].reshape(n_test, 1), test_predictions_and_y[:, n_parameters + i].reshape(n_test, 1)))
    parameter_predictions_and_y = parameter_predictions_and_y.reshape(n_test, 2)
    Parameter_Support.plot_results(subplots[i], parameter_predictions_and_y, test_score, parameter_letters[i])

test_figure.canvas.set_window_title('Test Results')
test_figure.legend()

train_figure.savefig(os.path.join(NEW_DIR_NAME, 'Test Results.png'), dpi=300)




plt.show()













































