from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMAGE_DIM = 200

def get_image(path):
    with Image.open(path) as image:
        image = image.resize((IMAGE_DIM, IMAGE_DIM))
        image = np.array(image) / 255
        return image.astype(np.float32)


def plot_results(plot, predictions_and_y, score, parameter, image_names=None):
    x = np.array(range(1, predictions_and_y.shape[0] + 1)).reshape((predictions_and_y.shape[0], 1))
    predictions_and_y = predictions_and_y[predictions_and_y[:, 1].argsort()] # Sort each row based on the second column, the actual values

    plot.scatter(x, predictions_and_y[:, 0], label='Predicted', color='firebrick', s=10)
    plot.scatter(x, predictions_and_y[:, 1], label='Actual', color='steelblue', s=10)
    plot.set_title('Predictions vs Actual Values, Score: ' + str(score))
    plot.set_xlabel('Index')
    plot.set_ylabel(parameter)

    print('x shape: ', x.shape)
    print('predictions and y shape: ', predictions_and_y.shape)
    print('image_names[5]: ', image_names[5])
    print('x[5]: ', x[5, 0])
    # print('predictions and y[0][5]: ', predictions_and_y[0][5])
    print('predictions and y[0, 5]: ', predictions_and_y[5, 0])


    if image_names is not None:
        for i, name in enumerate(image_names):
            plot.annotate(name, (x[i, 0], predictions_and_y[i, 0]))




