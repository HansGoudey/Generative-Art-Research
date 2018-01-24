from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
import Parameter_Support

IMAGE_DIR = "E:\\Generative-Art-Research\\Images\\Roundness_Test"


N_EPOCHS = 400
RUN_ID = 9

MODEL_DIR = 'results_' + str(N_EPOCHS) + '_epochs_' + 'id_' + str(RUN_ID)
TEST_NAME = os.path.basename(os.path.normpath(IMAGE_DIR))



# RETRIEVE THE PREVIOUSLY TRAINED MODEL FOR THE GIVEN ID NUMBER AND EPOCHS
model = load_model(os.path.join(MODEL_DIR, 'model_' + str(N_EPOCHS) + '_epochs_' + 'id_' + str(RUN_ID) + '.h5'))



# LOAD THE TEST DATA
image_names = os.listdir(IMAGE_DIR)
np.random.shuffle(image_names)
n_names = len(image_names)

x = np.array([Parameter_Support.get_image(os.path.join(IMAGE_DIR, name)) for name in image_names]).astype(np.float32)
y = np.array([float(name[5:7]) for name in image_names]).astype(np.float32)
y = y.reshape(y.shape[0], 1)



# MAKE PREDICTIONS FOR THE LOADED TEST IMAGES WITH THE LOADED MODEL
predictions = model.predict(x, batch_size=5)
predictions_and_y = np.hstack((predictions, y))
score = model.evaluate(x, y, batch_size=5)
np.savetxt(os.path.join(MODEL_DIR, TEST_NAME + '_results_' + str(N_EPOCHS) + '_epochs_' + 'id_' + str(RUN_ID) + '.csv'), predictions_and_y, delimiter=',')



# CREATE A PLOT OF THE RESULTS
figure, plot = plt.subplots(1, 1, figsize=(12, 6))
Parameter_Support.plot_results(plot, predictions_and_y, score, TEST_NAME, image_names)

plt.legend()

figure.savefig(os.path.join(MODEL_DIR, TEST_NAME + '_figure_' + str(N_EPOCHS) + '_epochs_' + 'id_' + str(RUN_ID) + '.png'), dpi=300)

plt.show()
































