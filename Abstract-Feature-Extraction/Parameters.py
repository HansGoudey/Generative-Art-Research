import errno
import os
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import ast
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
import Parameter_Models
from PIL import Image


def main():
	"""
	Contains the command line interface for the ParameterModel class
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument('operation', action='store', choices=['train', 'test', 'predict', 'info'], help='Which operation to do')
	parser.add_argument('directories', action='store', help='The directories to load images from')
	parser.add_argument('-l', '--model_to_load', action='store', dest='model_to_load', help='Previous model to load and use for training or predictions', default=None)
	parser.add_argument('-t', '--image_types', action='store', dest='image_types', help='Types (first 3 letters of file names) of images to train on. String of characters with no spaces', default=None)
	parser.add_argument('-n', '--run_id', action='store', dest='run_id', help='Name for operation. Supplying a name will create a new results folder with that name', default=None)
	parser.add_argument('-e', '--epochs', action='store', dest='n_epochs', type=int, help='Number of epochs to use for training and traintest operations', default=1)
	parser.add_argument('-m', '--parameter_map', action='store', dest='parameter_map', help='Map one parameter to another with a dictionary', default='{}')

	args = parser.parse_args()

	if args.operation == 'train':
		if args.run_id is None and args.model_to_load is None:  # The run will have a new folder created for it and it needs a new name
			args.run_id = args.operation + ' ' + str(args.n_epochs) + ' epochs ' + 'from ' + os.path.basename(os.path.normpath(args.directories[0]))

		model = ParameterModel(args.model_to_load, args.run_id)

		if hasattr(args, 'image_types'):
			model.set_image_types(args.image_types)

		model.train_operation(ast.literal_eval(args.directories), args.n_epochs, args.image_types, ast.literal_eval(args.parameter_map))

	elif args.operation == 'test':
		print('Test Operation')
		if args.model_to_load is None:  # This is a completely new model and a new training run
			sys.exit('A test operation must load a model')

		print('MODEL TO LOAD', args.model_to_load)
		model = ParameterModel(args.model_to_load, args.run_id)

		model.test_operation(args.directories, args.image_types, ast.literal_eval(args.parameter_map), args.model_to_load)
	elif args.operation == 'predict':
		model = ParameterModel(args.model_to_load, args.run_id)
	elif args.operation == 'info':
		pass

	plt.show()


class ParameterModel:
	"""
	Creating a new model and training on it
		model = ParameterModel(run_id, None)
		model.train_operation(directories, epochs, image_types=image_types, parameter_map=parameter_map)

	Training an existing model without duplicating/creating a new folder
		model = ParameterModel(None, model_to_load)
		model.train_operation(directories, epochs, image_types=image_types, parameter_map=parameter_map)

	Training an existing model and creating a new folder for the changed model
		model = ParameterModel(run_id, model_to_load)
		model.train_operation(directories, epochs, image_types=image_types, parameter_map=parameter_map)

	Testing an existing model on a data set
		model = ParameterModel(run_id (optional), model_to_load)
		model.test_op(directories, image_types=image_types, parameter_map=parameter_map)

	Predicting values for a set of unlabeled images
		model = ParameterModel(run_id (optional), model_to_load)
		model.predict_op(directories)

	Getting info on an existing ParameterModel Object that has been saved to a directories
		model = ParameterModel(None, model_to_load)
		model.get_info()
	"""

	model = None
	results_dir = ''

	trained_parameter_map = {}  # Dictionary mapping of equivalent parameters. {'f':'r'} means 'f' should be equivalent to 'r' for training
	trained_image_types = []  # A list of the first three identifier letters at the beginning of the image files to only train on those types
	trained_parameters = None
	n_trained_parameters = 1
	loaded_model = None
	trained_epochs = 0

	test_margin = 10

	batch_size = 300
	image_dim = 0  # TODO: Add support for rectangular images

	x_train = []
	y_train = []
	x_test = []
	y_test = []

	train_predictions = []
	test_predictions = []

	def __init__(self, model_to_load, run_id):
		# Loads data and creates model by either loading an existing model or creating a new one

		if model_to_load:
			self.loaded_model = os.path.basename(os.path.normpath(model_to_load))
			self.retrieve_model(model_to_load)

		if run_id:
			self.make_results_directory(run_id)  # Overrides the self.results_dir value from retrieve model
			print('Created results directory:', self.results_dir)
		else:
			# Existing results directory should be reused because no run_id has been provided
			# run_id must be provided if model_to_load is not provided, because a new model is created TODO: Check this
			print('Results in directory:', self.results_dir)


	def train_operation(self, image_dirs, epochs, image_types, parameter_map):
		# Loads and trains on data and saves/shows result data and plots

		print('Loading Data')
		self.load_train_and_test_data(image_dirs, image_types, parameter_map)

		if self.model is None:
			print('Creating Model')
			self.create_model(self.n_trained_parameters)  # n_trained_parameters will be filled because data loading happens just before

		history = self.train(epochs)  # Fills self.train_predictions
		self.plot_loss_history(history)
		train_scores = margin_metric(self.test_margin, self.train_predictions, self.y_train)
		test_scores = self.test()
		print('Test Score: ', test_scores)
		self.plot_against_y(self.train_predictions, self.y_train, 'Train Predictions vs Actual Values', train_scores)
		self.plot_against_y(self.test_predictions, self.y_test, 'Test Predictions vs Actual Values', test_scores)

		self.save_model_and_params()
		self.save_training_description(image_dirs)

	def test_operation(self, image_dirs, image_types, parameter_map, loaded_model):
		# Assumes model has already been loaded when the ParameterObject object was created

		print('Loading Data')
		self.load_test_data(image_dirs, image_types, parameter_map)
		self.results_dir = loaded_model

		test_scores = self.test()
		print('Test Score: ', test_scores)
		self.plot_against_y(self.test_predictions, self.y_test, os.path.basename(os.path.normpath(image_dir)) + ' Predictions vs Values', test_scores)

	def predict_operation(self, image_dirs, model_to_load):
		# For making a set of predictions from unlabeled data


		print('Loading Data')
		self.load_only_x(image_dirs)


	def retrieve_model(self, model_dir):
		try:
			self.model = load_model(os.path.join(model_dir, 'Model.h5'))
		except (ImportError, ValueError):
			sys.exit('Error importing model.h5 file.' + os.path.join(model_dir, 'Model.h5') + 'No such file, or incompatible')

		with open(os.path.join(model_dir, 'Parameters.pickle'), 'rb') as parameters_file:
			parameters = pickle.load(parameters_file)
			self.results_dir                = parameters['results_dir']
			self.trained_parameter_map      = parameters['parameter_map']
			self.trained_image_types        = parameters['image_types']
			self.trained_parameters         = parameters['trained_parameters']
			self.n_trained_parameters       = len(self.trained_parameters)
			self.batch_size                 = parameters['batch_size']


	def make_results_directory(self, run_id):
		results_dir = str(run_id) + ' Results'
		try:
			os.makedirs(results_dir)
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				raise

		self.results_dir = results_dir


	def load_train_and_test_data(self, image_dirs, image_types, parameter_map):
		x, y = self.load_data(image_dirs, image_types, parameter_map, True)

		test_split = int(x.shape[0] * 0.8)

		self.x_train, self.x_test = np.array_split(x, [test_split])
		self.y_train, self.y_test = np.array_split(y, [test_split])

	def load_test_data(self, image_dirs, image_types, parameter_map):
		x, y = self.load_data(image_dirs, image_types, parameter_map, True)

		self.x_test = x
		self.y_test = y

	def load_only_x(self, image_dirs):
		# For loading unlabelled data.
		x = self.load_data(image_dirs, None, None, True)

		self.x_test = x

	def load_data(self, image_dirs, image_types, parameter_map, load_y):  # TODO: Add support for loading images from multiple directories
		"""
		Gets x and y values for images with types matching values in image_types in the specified directories.
		Maps parameters

		Fills the following instance variables:
			trained_parameters
			n_trained_parameters
			x_train, x_test, y_train, y_test
			image_dim
		"""
		# Each image name should be the combination of its immediate folder and the file name of the image
		image_names = []
		for image_dir in image_dirs:  # Get image names from all provided directories
			image_names += [os.path.join(os.path.basename(os.path.dirname(image_dir)), name) for name in os.listdir(image_dir)]  # Add the name of the folder the image is from to every image name
		np.random.shuffle(image_names)
		
		image_dir_dir = os.path.dirname(os.path.dirname(image_dirs[0]))  # The directory that holds all of the image directories. This assumes all of the directories live in a common folder.
		self.get_image_dim(os.path.join(image_dir_dir, image_names[1]))

		# Remove the images types that aren't wanted for training if some are specified with a list
		if image_types is not None:
			image_names = [name for name in image_names if any(kind in name for kind in image_types)]

		n_names = len(image_names)

		x = np.array([self.get_image(os.path.join(image_dir_dir, name)) for name in image_names])
		x = x.reshape((n_names, self.image_dim, self.image_dim, 1)).astype(np.float32)

		if not load_y:  # For unlabeled data, the function should return before it tries to gather labels
			return x

		n_image_parameters = image_names[0].count('-') - 1
		y = np.ones(shape=(n_names, n_image_parameters), dtype=np.float32) * -100  # Initialize the array to the middle value in case some file names are missing data

		temp_image_names = list(image_names)  # Duplicate list so the original names aren't changed in case they need to be used

		# Map the parameters in the parameter map dictionary
		for i in range(len(temp_image_names)):
			for letter in parameter_map:
				if '-' + letter in temp_image_names[i]:
					temp_image_names[i] = temp_image_names[i].replace('-' + letter, '-' + parameter_map[letter])
		self.trained_parameter_map = parameter_map

		parameter_indexes = [m.start() + 1 for m in re.finditer('-', temp_image_names[0])][0:-1]  # Don't use the last '-' in the name. It's before the last number, not a parameter
		self.trained_parameters = [temp_image_names[0][i] for i in parameter_indexes]  # Get the parameters from the first image. Assumes all images have consistent parameters
		self.n_trained_parameters = len(self.trained_parameters)

		# Build y values
		for i_name in range(len(temp_image_names)):
			for i_letter in range(len(self.trained_parameters)):
				letter = self.trained_parameters[i_letter]
				index_in_name = temp_image_names[i_name].find('-' + letter)
				if index_in_name == -1:
					# print(image_names[i_name], 'is missing the', letter, 'parameter')
					continue

				value = temp_image_names[i_name][index_in_name + 2:index_in_name + 4]
				y[i_name, i_letter] = value

		return x, y

		# self.x_train, self.x_test = np.array_split(x, [test_split])
		# self.y_train, self.y_test = np.array_split(y, [test_split])
		#
		# # FIND IMAGE NAMES THAT AREN'T GIVING VALUES
		# for i in range(len(temp_image_names)):
		# 	if y[i, 0] == -100:
		# 		print('Value not assigned, name:', temp_image_names[i])

	def get_image_dim(self, image_path):
		with Image.open(image_path) as image:
			width, height = image.size
		assert (width == height), 'Images should be square'
		self.image_dim = width

	def get_image(self, path):
		with Image.open(path) as image:
			image = np.array(image) / 255
			return image[:, :, 0].reshape((self.image_dim, self.image_dim, 1)).astype(np.float32)


	def create_model(self, n_parameters):
		self.model = Parameter_Models.more_conv_multiple(self.image_dim, n_parameters)


	def save_training_description(self, image_dirs):
		with open(os.path.join(self.results_dir, 'Model Summary.txt'), 'w+') as summary_file:
			summary_file.write('Parameters: ')
			for letter in self.trained_parameters:
				summary_file.write(letter + ', ')
			summary_file.write('\n' + 'Trained from ')
			for image_dir in image_dirs:
				summary_file.write(image_dir + ', ')
			summary_file.write('\n')
			if self.loaded_model:
				summary_file.write('Loaded from previously trained model in ' + self.loaded_model + '\n')
			if self.trained_image_types is None:
				summary_file.write('Trained on all image types')
			else:
				summary_file.write('Valid image types: ')
				for kind in self.trained_image_types:
					summary_file.write(kind + ', ')
			summary_file.write('\n' + 'Trained for ' + str(self.trained_epochs) + ' epochs.' + '\n')
			summary_file.write('Tested with a margin of ' + str(self.test_margin) + ' points.' + '\n')
			summary_file.write('Images had dimension ' + str(self.image_dim) + ' pixels, square.' + '\n')

			summary_file.write('\n')

			self.model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
			
			plot_model(self.model, to_file=os.path.join(self.results_dir, 'Model Plot.png'), show_shapes=True, show_layer_names=True)

	def save_model_and_params(self):
		self.model.save(os.path.join(self.results_dir, 'Model.h5'))

		to_save = {'results_dir': self.results_dir, 'parameter_map': self.trained_parameter_map, 'image_types': self.trained_image_types, 'trained_parameters': self.trained_parameters, 'batch_size': self.batch_size}
		with open(os.path.join(self.results_dir, 'Parameters.pickle'), 'wb') as parameter_file:
			pickle.dump(to_save, parameter_file)


	def train(self, n_epochs):
		# Trains on images that are already loaded into the object's instance variables
		# If n_epochs is less than 1, the model will stop training when the model has not improved in the absolute value of that number os epochs

		if n_epochs < 0:
			converge_monitor = EarlyStopping(patience=abs(n_epochs))
			history = self.model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=self.batch_size, callbacks=[converge_monitor])
		else:
			history = self.model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=self.batch_size)

		self.train_predictions = self.model.predict(self.x_train, batch_size=self.batch_size)
		np.clip(self.train_predictions, 0, 100, out=self.train_predictions)

		self.trained_epochs = n_epochs
		return history


	def plot_loss_history(self, history):
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

		np.savetxt(os.path.join(self.results_dir, 'Loss History.csv'), loss_history, delimiter=',', header=(','.join(self.trained_parameters) + ',') * 2)

		figure.canvas.set_window_title('Loss History')
		figure.savefig(os.path.join(self.results_dir, 'Loss History.png'), dpi=300)


	def test(self):
		self.test_predictions = self.model.predict(self.x_test, batch_size=self.batch_size)
		np.clip(self.test_predictions, 0, 100, out=self.test_predictions)

		np.savetxt(os.path.join(self.results_dir, 'Test Results.csv'), np.hstack((self.test_predictions, self.y_test)), delimiter=',', header=(','.join(self.trained_parameters) + ',') * 2)

		test_scores = margin_metric(self.test_margin, self.test_predictions, self.y_test)

		return test_scores


	def plot_predictions(self, predictions, title):
		train_figure, subplots = plt.subplots(1, self.n_trained_parameters, figsize=(6 * self.n_trained_parameters, 6))  # Create a subplot for each parameter

		for i in range(self.n_trained_parameters):
			n_train = predictions.shape[0]

			predictions = predictions.reshape(n_train, 2)
			plot_results(subplots[i], predictions, self.trained_parameters[i], title)  # TODO Fix this, a new function probably needed

	def plot_against_y(self, predictions, y, title, score):
		train_figure, subplots = plt.subplots(1, self.n_trained_parameters, figsize=(6 * self.n_trained_parameters, 6))  # Create a subplot for each parameter

		predictions_and_y = np.hstack((predictions, y))
		if self.n_trained_parameters == 1:
			plot_results(subplots, predictions_and_y, score, self.trained_parameters)
		else:
			for i in range(self.n_trained_parameters):
				n_train = predictions_and_y.shape[0]
				parameter_predictions_and_y = np.hstack((predictions_and_y[:, i].reshape(n_train, 1), predictions_and_y[:, self.n_trained_parameters + i].reshape(n_train, 1)))
				parameter_predictions_and_y = parameter_predictions_and_y.reshape(n_train, 2)
				plot_results(subplots[i], parameter_predictions_and_y, score[i], self.trained_parameters[i])

		train_figure.canvas.set_window_title(title)
		train_figure.legend()

		train_figure.savefig(os.path.join(self.results_dir, title + '.png'), dpi=300)


	def set_parameter_map(self, new_parameter_map):
		# The parameter map is used equivocate one image parameter to another for training.
		# For example, to test how similar noise and roundness are, you could map f (noise) to r (roundness) with the map {'f':'r'}
		self.trained_parameter_map = new_parameter_map

	def set_image_types(self, image_types):
		self.trained_image_types = image_types

	def set_margin(self, margin):
		self.test_margin = margin


def margin_metric(margin, x, y):
	# Returns ratio of x that is within the provided margin of y
	# x and y are numpy arrays. They can have multiple columns, one for each parameter
	return ((x < y + margin) & (x > y - margin)).sum(axis=0)/x.shape[0]


def plot_results(plot, predictions_and_y, score, parameter, image_names=None):
	x = np.array(range(1, predictions_and_y.shape[0] + 1)).reshape((predictions_and_y.shape[0], 1))

	# Sort the predictions and y values by the y values
	indexes_sorted_by_y = predictions_and_y[:, 1].argsort()
	predictions_and_y = predictions_and_y[indexes_sorted_by_y]

	plot.scatter(x, predictions_and_y[:, 0], label='Predicted', color='firebrick', s=10)
	plot.scatter(x, predictions_and_y[:, 1], label='Actual', color='steelblue', s=10)
	plot.set_title('Score: ' + str(score))
	plot.set_xlabel('Index')
	plot.set_ylabel(parameter)
	plot.grid()


if __name__ == '__main__':
	main()
