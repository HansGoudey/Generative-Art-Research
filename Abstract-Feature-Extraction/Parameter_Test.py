from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
import Parameter_Support

IMAGE_DIR = "E:\\Generative-Art-Research\\Images\\Noise"


N_EPOCHS = 300
RUN_ID = 18
MODEL_DIR = 'Run ' + str(RUN_ID) + ' Results ' + str(N_EPOCHS) + ' Epochs'

TEST_NAME = os.path.basename(os.path.normpath(IMAGE_DIR))



# RETRIEVE THE PREVIOUSLY TRAINED MODEL FOR THE GIVEN ID NUMBER AND EPOCHS
model = load_model(os.path.join(MODEL_DIR, 'Model.h5'))



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
np.savetxt(os.path.join(MODEL_DIR, 'Test Predictions ' + TEST_NAME + ' Results.csv'), predictions_and_y, delimiter=',')



# CREATE A PLOT OF THE RESULTS
image_names = np.array(image_names)
image_names = image_names.reshape(len(image_names), 1)
indexes_sorted_by_y = predictions_and_y[:, 1].argsort()
image_names = image_names[indexes_sorted_by_y]

x_in_plot = np.array(range(1, predictions_and_y.shape[0] + 1)).reshape((predictions_and_y.shape[0], 1))
figure, plot = plt.subplots(1, 1, figsize=(12, 6))

def on_plot_click(event):
    ex, ey = event.xdata, event.ydata
    print('click location: (', ex, ', ', ey, ')')

    if ex is not None:
        ax = figure.gca()
        ax = plot
        dx = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        dy = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])

        close_image_names = []
        for i in range(x_in_plot.shape[0]):
            if ex-dx < x_in_plot[i, 0] < ex+dx and ey-dy < predictions[i, 0] < ey+dy:
                close_image_names.append(image_names[i, 0])
        if close_image_names:
            "Opening clicked image"
            Parameter_Support.show_image(os.path.join(IMAGE_DIR, np.random.choice(close_image_names)))

cid = figure.canvas.mpl_connect('button_press_event', on_plot_click)
Parameter_Support.plot_results(plot, predictions_and_y, score, TEST_NAME, image_names)

plt.legend()

figure.savefig(os.path.join(MODEL_DIR, 'Test Predictions ' + TEST_NAME + ' Results.png'), dpi=300)

plt.show()
































