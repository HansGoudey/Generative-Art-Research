from keras.models import load_model
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import Parameter_Support

IMAGE_DIR = "C:\\Users\\Hans Goudey\\Downloads\\Google Images\\Round"
IMAGE_DIR = "E:\\Generative-Art-Research\\Images\\Messy Round Rects"

MODEL_DIR = "E:\\Generative-Art-Research\\Abstract-Feature-Extraction\\Run 302 Results 1 Epochs"


TEST_NAME = os.path.basename(os.path.normpath(IMAGE_DIR))




########################################################################################################################
# RETRIEVE THE PREVIOUSLY TRAINED MODEL AND ITS PARAMETER OUTPUT LABELS
########################################################################################################################
model = load_model(os.path.join(MODEL_DIR, 'Model.h5'))
with open(os.path.join(MODEL_DIR, 'Parameters.pickle'), 'rb') as parameters_file:
    parameters = pickle.load(parameters_file)
n_parameters = len(parameters)




########################################################################################################################
# LOAD THE IMAGE DATA
########################################################################################################################
image_names = os.listdir(IMAGE_DIR)
n_names = len(image_names)

x = np.array([Parameter_Support.get_new_image(os.path.join(IMAGE_DIR, name)) for name in image_names]).astype(np.float32)




########################################################################################################################
# MAKE AND SAVE PREDICTIONS FOR THE LOADED TEST IMAGES WITH THE LOADED MODEL
########################################################################################################################
predictions = model.predict(x, batch_size=100)
image_names = np.array(image_names).reshape(n_names, 1)

names_and_predictions = np.hstack((image_names, predictions))

np.savetxt(os.path.join(MODEL_DIR, 'Predictions ' + TEST_NAME + ' Results.csv'), names_and_predictions, delimiter=',', fmt="%s")




########################################################################################################################
# CREATE A PLOT OF THE RESULTS
########################################################################################################################
image_names = np.array(image_names).reshape(len(image_names), 1)

x_in_plot = np.array(range(1, n_names + 1)).reshape((n_names, 1))
figure, plot = plt.subplots(1, figsize=(8, 6))

def on_plot_click(event):
    ex, ey = event.xdata, event.ydata

    if ex is not None:
        ax = figure.gca()
        ax = plot
        dx = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        dy = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])

        close_image_names = []
        for i in range(n_names):
            for j in range(n_parameters):
                if ex-dx < x_in_plot[i, j] < ex+dx and ey-dy < predictions[i, j] < ey+dy:
                    close_image_names.append(image_names[i, 0])
        if close_image_names:
            Parameter_Support.show_image(os.path.join(IMAGE_DIR, np.random.choice(close_image_names)))

cid = figure.canvas.mpl_connect('button_press_event', on_plot_click)

colors = ['#6495ED','#00FFFF','#7FFFD4','#F0FFFF','#F5F5DC','#FFE4C4','#000000','#FFEBCD','#0000FF','#8A2BE2','#A52A2A','#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50','#0AEBD7',
          '#FFF8DC','#DC143C','#00FFFF','#00008B','#008B8B','#B8860B','#A9A9A9','#006400','#BDB76B','#8B008B','#556B2F','#FF8C00','#9932CC','#8B0000','#E9967A','#8FBC8F','#483D8B']
np.random.shuffle(colors)
for i in range(n_parameters):
    plot.scatter(x_in_plot, predictions[:, i], label=parameters[i], color=colors[i], s=10)


# for i, name in enumerate(image_names):
#     plot.annotate(name, (x_in_plot[i, 0], predictions[i, 0]))

plt.legend()

figure.savefig(os.path.join(MODEL_DIR, 'Predictions ' + TEST_NAME + ' Results.png'), dpi=300)

plt.show()






























