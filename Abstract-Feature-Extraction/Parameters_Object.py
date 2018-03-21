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
import Parameter_Models
import Parameter_Support


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('operation', action='store', choices=['train', 'test', 'predict', 'info'], help='Which operation to do')
	parser.add_argument('directory', action='store', help='The directory to load images from')
	parser.add_argument('-l', '--model_to_load', action='store', dest='model_to_load', help='Previous model to load and use for training or predictions', default=None)
	parser.add_argument('-t', '--image_types', action='store', dest='image_types', help='Types (first 3 letters of file names) of images to train on. String of characters with no spaces', default=None)
	parser.add_argument('-n', '--run_id', action='store', dest='run_id', help='Name for operation. Supplying a name will create a new results folder with that name', default=None)
	parser.add_argument('-e', '--epochs', action='store', dest='n_epochs', type=int, help='Number of epochs to use for training and traintest operations', default=1)
	parser.add_argument('-m', '--parameter_map', action='store', dest='parameter_map', help='Map one parameter to another with a dictionary', default='{}')

	args = parser.parse_args()

	if args.operation == 'train':
		if args.run_id is None:  # The run will have a new folder created for it and it needs a new name
			# TODO: Make this only happen when model_to_load isn't provided
			args.run_id = args.operation + ' ' + str(args.n_epochs) + ' epochs ' + 'from ' + os.path.basename(os.path.normpath(args.directory))

		model = ParameterModel(args.model_to_load, args.run_id)

		if hasattr(args, 'image_types'):
			model.set_image_types(args.image_types)

		model.train_operation(args.directory, args.n_epochs, args.image_types, ast.literal_eval(args.parameter_map))

	elif args.operation == 'test':
		print('Test Operation')
		if args.model_to_load is None:  # This is a completely new model and a new training run
			sys.exit('A test operation must load a model')

		print('MODEL TO LOAD', args.model_to_load)
		model = ParameterModel(args.model_to_load, args.run_id)

		model.test_operation(args.directory, args.image_types, ast.literal_eval(args.parameter_map), args.model_to_load)
	elif args.operation == 'predict':
		pass
	elif args.operation == 'info':
		pass

	plt.show()


"""
HOW THE CLASS SHOULD BE USED:

Creating a new model and training on it
	model = ParameterModel(run_id, None)
	model.train_operation(directory, epochs, image_types=image_types, parameter_map=parameter_map)
	
Training an existing model without duplicating/creating a new folder
	model = ParameterModel(None, model_to_load)
	model.train_operation(directory, epochs, image_types=image_types, parameter_map=parameter_map)
	
Training an existing model and creating a new folder for the changed model
	model = ParameterModel(run_id, model_to_load)
	model.train_operation(directory, epochs, image_types=image_types, parameter_map=parameter_map)
	
Testing an existing model on a data set
	model = ParameterModel(run_id (optional), model_to_load)
	model.test_op(directory, image_types=image_types, parameter_map=parameter_map)
	
Predicting values for a set of unlabeled images
	model = ParameterModel(run_id (optional), model_to_load)
	model.predict_op(directory)
	
Getting info on an existing ParameterModel Object that has been saved to a directory
	model = ParameterModel(None, model_to_load)
	model.get_info()
"""
class ParameterModel:

	model = None
	results_dir = ''

	trained_parameter_map = {}  # Dictionary mapping of equivalent parameters. {'f':'r'} means 'f' should be equivalent to 'r' for training
	trained_image_types = []  # A list of the first three identifier letters at the beginning of the image files to only train on those types
	trained_parameters = []
	n_trained_parameters = 1
	loaded_model = None
	trained_epochs = 0  # TODO: Output this in the model summary

	batch_size = 300

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


	def train_operation(self, image_dir, epochs, image_types, parameter_map):
		# Loads and trains on data and saves/shows result data and plots

		print('Loading Data')
		self.load_train_and_test_data(image_dir, image_types, parameter_map)

		if self.model is None:
			self.create_model(self.n_trained_parameters)  # n_trained_parameters will be filled because data loading happens just before

		history = self.train(epochs)
		self.plot_loss_history(history)
		test_score = self.test()
		train_score = self.model.evaluate(self.x_train, self.y_train, batch_size=self.batch_size)
		self.plot_against_y(self.train_predictions, self.y_train, 'Train Predictions vs Actual Values', train_score)
		self.plot_against_y(self.test_predictions, self.y_test, 'Test Predictions vs Actual Values', test_score)

		self.save_model_and_params()
		self.save_training_description(image_dir, image_types, self.loaded_model)

	def test_operation(self, image_dir, image_types, parameter_map, loaded_model):
		# Assumes model has already been loaded when the ParameterObject object was created

		print('Loading Data')
		self.load_test_data(image_dir, image_types, parameter_map)
		self.results_dir = loaded_model

		test_score = self.test()
		self.plot_against_y(self.test_predictions, self.y_test, os.path.basename(os.path.normpath(image_dir)) + ' Predictions vs Values', test_score)


	def retrieve_model(self, model_dir):
		try:
			self.model = load_model(os.path.join(model_dir, 'Model.h5'))
		except (ImportError, ValueError) as e:
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
		print('Making directory', results_dir)
		try:
			os.makedirs(results_dir)
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				raise

		self.results_dir = results_dir


	def load_train_and_test_data(self, image_dir, image_types, parameter_map):
		x, y = self.load_data(image_dir, image_types, parameter_map)

		test_split = int(x.shape[0] * 0.8)

		self.x_train, self.x_test = np.array_split(x, [test_split])
		self.y_train, self.y_test = np.array_split(y, [test_split])

	def load_test_data(self, image_dir, image_types, parameter_map):
		x, y = self.load_data(image_dir, image_types, parameter_map)

		self.x_test = x
		self.y_test = y

	def load_data(self, image_dir, image_types, parameter_map):
		"""
		Gets x and y values for images with types matching values in image_types in the specified directory.
		Maps parameters

		Fills the following instance variables:
			trained_parameters
			n_trained_parameters
			x_train, x_test, y_train, y_test
		"""

		image_names = os.listdir(image_dir)
		np.random.shuffle(image_names)

		# Remove the images types that aren't wanted for training if some are specified with a list
		if image_types is not None:
			image_names = [name for name in image_names if any(kind in name for kind in image_types)]

		n_names = len(image_names)
		print(n_names, 'Images')

		x = np.array([Parameter_Support.get_image(os.path.join(image_dir, name)) for name in image_names])
		x = x.reshape((n_names, Parameter_Support.IMAGE_DIM, Parameter_Support.IMAGE_DIM, 1)).astype(np.float32)

		n_image_parameters = image_names[0].count('-') - 1
		y = np.ones(shape=(n_names, n_image_parameters), dtype=np.float32) * -100  # Initialize the array to the middle value in case some file names are missing data

		temp_image_names = list(image_names)  # Duplicate list so the original names aren't changed in case they need to be used

		# Map the parameters in the parameter map dictionary
		print('Parameter Map:', parameter_map)
		for i in range(len(temp_image_names)):
			for letter in parameter_map:
				if '-' + letter in temp_image_names[i]:
					temp_image_names[i] = temp_image_names[i].replace('-' + letter, '-' + parameter_map[letter])
		self.trained_parameter_map = parameter_map

		parameter_indexes = [m.start() + 1 for m in re.finditer('-', temp_image_names[0])][0:-1]  # Don't use the last '-' in the name. It's before the last number, not a parameter
		self.trained_parameters = [temp_image_names[0][i] for i in parameter_indexes]  # Get the parameters from the first image. Assumes all images have consistent parameters
		self.n_trained_parameters = len(self.trained_parameters)
		print('parameter letters', self.trained_parameters)

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


	def create_model(self, n_parameters):
		print('Creating Model')
		self.model = Parameter_Models.more_conv_multiple(Parameter_Support.IMAGE_DIM, n_parameters)


	def save_training_description(self, image_dir, image_types, previous_model):
		with open(os.path.join(self.results_dir, 'Model Summary.txt'), 'w+') as summary_file:
			summary_file.write('\nParameters: ')
			for letter in self.trained_parameters:
				summary_file.write(letter + ', ')
			summary_file.write('Trained from ' + image_dir + '\n')
			if previous_model:
				summary_file.write('Loaded from previously trained model in ' + previous_model + '\n')
			if image_types is None:
				summary_file.write('Trained on all image types')
			else:
				summary_file.write('Valid image types: ')
				for kind in image_types:
					summary_file.write(kind + ', ')

			summary_file.write('\n\n')

			self.model.summary(print_fn=lambda x: summary_file.write(x + '\n'))

	def save_model_and_params(self):
		self.model.save(os.path.join(self.results_dir, 'Model.h5'))

		to_save = {'results_dir': self.results_dir, 'parameter_map': self.trained_parameter_map, 'image_types': self.trained_image_types, 'trained_parameters': self.trained_parameters, 'batch_size': self.batch_size}
		with open(os.path.join(self.results_dir, 'Parameters.pickle'), 'wb') as parameter_file:
			pickle.dump(to_save, parameter_file)


	def train(self, n_epochs):
		history = self.model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=self.batch_size)

		self.train_predictions = self.model.predict(self.x_train, batch_size=self.batch_size)

		return history


	# def evaluate_against_y(self):
	# 	train_predictions_and_y = np.hstack((self.train_predictions, self.y_train))
	# 	print('train predictions and y shape', train_predictions_and_y.shape)
	#
	# 	np.savetxt(os.path.join(self.results_dir, 'Train Results.csv'), train_predictions_and_y, delimiter=',', header=(','.join(self.trained_parameters) + ',') * 2)
	#
	# 	train_score = self.model.evaluate(self.x_train, self.y_train, batch_size=self.batch_size)
	# 	print('Train Score: ', train_score)
	#
	# 	return train_score


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
		test_predictions_and_y = np.hstack((self.test_predictions, self.y_test))

		np.savetxt(os.path.join(self.results_dir, 'Test Results.csv'), test_predictions_and_y, delimiter=',', header=(','.join(self.trained_parameters) + ',') * 2)

		test_scores = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)

		print('Test Score: ', test_scores)

		return test_scores


	def plot_predictions(self, predictions, title):
		train_figure, subplots = plt.subplots(1, self.n_trained_parameters, figsize=(6 * self.n_trained_parameters, 6))  # Create a subplot for each parameter

		for i in range(self.n_trained_parameters):
			n_train = predictions.shape[0]

			predictions = predictions.reshape(n_train, 2)
			Parameter_Support.plot_results(subplots[i], predictions, self.trained_parameters[i])  # TODO

	def plot_against_y(self, predictions, y, title, score):
		train_figure, subplots = plt.subplots(1, self.n_trained_parameters, figsize=(6 * self.n_trained_parameters, 6))  # Create a subplot for each parameter

		predictions_and_y = np.hstack((predictions, y))
		if self.n_trained_parameters == 1:
			Parameter_Support.plot_results(subplots, predictions_and_y, score, self.trained_parameters)
		else:
			for i in range(self.n_trained_parameters):
				n_train = predictions_and_y.shape[0]
				parameter_predictions_and_y = np.hstack((predictions_and_y[:, i].reshape(n_train, 1), predictions_and_y[:, self.n_trained_parameters + i].reshape(n_train, 1)))
				parameter_predictions_and_y = parameter_predictions_and_y.reshape(n_train, 2)
				Parameter_Support.plot_results(subplots[i], parameter_predictions_and_y, score[i], self.trained_parameters[i])

		train_figure.canvas.set_window_title(title)
		train_figure.legend()

		train_figure.savefig(os.path.join(self.results_dir, title + '.png'), dpi=300)


	def set_parameter_map(self, new_parameter_map):
		# The parameter map is used equivocate one image parameter to another for training.
		# For example, to test how similar noise and roundness are, you could map f (noise) to r (roundness) with the map {'f':'r'}
		self.trained_parameter_map = new_parameter_map

	def set_image_types(self, image_types):
		self.trained_image_types = image_types


if __name__ == '__main__':
	main()


