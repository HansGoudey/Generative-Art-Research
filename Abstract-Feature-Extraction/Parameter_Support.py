from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

IMAGE_DIM = 200

def get_image(path):
    with Image.open(path) as image:
        image = image.resize((IMAGE_DIM, IMAGE_DIM))
        image = np.array(image) / 255
        return image.astype(np.float32)


def plot_results(plot, predictions_and_y, score, parameter, image_names=None):
    x = np.array(range(1, predictions_and_y.shape[0] + 1)).reshape((predictions_and_y.shape[0], 1))

    # Sort the predictions and y values by the y values
    indexes_sorted_by_y = predictions_and_y[:, 1].argsort()
    predictions_and_y = predictions_and_y[indexes_sorted_by_y]

    plot.scatter(x, predictions_and_y[:, 0], label='Predicted', color='firebrick', s=10)
    plot.scatter(x, predictions_and_y[:, 1], label='Actual', color='steelblue', s=10)
    plot.set_title('Predictions vs Actual Values, Score: ' + str(score))
    plot.set_xlabel('Index')
    plot.set_ylabel(parameter)
    plot.grid()

    # if image_names is not None:
    #     image_names = image_names[indexes_sorted_by_y]
    #     for i, name in enumerate(image_names):
    #         plot.annotate(name, (x[i, 0], predictions_and_y[i, 0]))


def show_image(file_name):
    figure = plt.figure(frameon=False, figsize=(4, 4))
    figure.canvas.set_window_title(file_name)
    plot = figure.add_subplot(111)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plot.axis("off")
    plot.set_title(os.path.basename(os.path.normpath(file_name)))  # Get the file name from the path

    image = mpimg.imread(file_name)
    plot.imshow(image)
    plt.show()


