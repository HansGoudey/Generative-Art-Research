import errno
import os
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from keras.models import load_model
import Parameter_Models
import Parameter_Support



def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('operation', action='store', choices=['train', 'test', 'predict'], help='Which operation to do')
	parser.add_argument('directory', action='store', dest='image_dir', help='The directory to load images from')
	parser.add_argument('-l', '--model_to_load', action='store', dest='model_to_load', help='Previous model to load and use for training or predictions')
	parser.add_argument('-t', '--image_types', action='store', dest='image_types', help='Types (first 3 letters of file names) of images to train on. String of characters with no spaces')
	parser.add_argument('-n', '--operation_name', action='store', dest='run_id', help='Name for operation. Supplying a name will create a new results folder with that name')
	parser.add_argument('-e', '--epochs', action='store', dest='n_epochs', type=int, help='Number of epochs to use for training and traintest operations', default=1)

	args = parser.parse_args()
	print('args:', args)

	if args.operation is 'train':
		if args.image_dir is None:
			sys.exit('Supply a directory of images to train on')

		if args.model_to_load is None:  # This is a completely new model and a new training run
			run_id = args.operation + ' ' + args.epochs + ' epochs ' + 'from ' + os.path.basename(os.path.normpath(args.image_dir))
			model = ParameterModel(args.model_to_load, args.image_dir, run_id=run_id)
			history = model.train(args.epochs)
			model.test()
			model.plot_loss_history(history)

	elif args.operation is 'test':
		pass
	elif args.operation is 'predict':
		pass

	plt.show()



class ParameterModel:

	model = []
	results_dir = ''
	parameter_map = {'f':'r'}  # Dictionary mapping of equivalent parameters. {'f':'r'} means 'f' should be equivalent to 'r' for training
	image_types = []
	trained_parameters = []
	n_parameters = 0
	batch_size = 300

	x_train = []
	y_train = []
	x_test = []
	y_test = []

	train_predictions = []
	test_predictions = []



	IMAGE_DIR = "E:\\Generative-Art-Research\\Images\\Messy Round Rects"
	IMAGE_TYPES = ['mes', 'rdr', 'rns', 'rsc', 'img', 'rmr']  # Add a list of the first three identifier letters at the beginning of the image files to only train on those types
	MODEL_TO_LOAD = "Run 301 Results 200 Epochs"  # Add the name of the folder containing the model to load and start the training with  'Run 18 Results 300 Epochs'  #



	def __init__(self, model_to_load, image_dir, run_id=None, image_types=None):
		# Loads data and creates model by either loading an existing model or creating a new one

		self.load_data(image_dir, image_types)

		if not model_to_load:
			self.create_model(self.n_parameters)
		else:
			self.retrieve_model(model_to_load)

		if run_id:
			self.make_results_directory(run_id)



	def retrieve_model(self, model_dir):
		try:
			self.model = load_model(os.path.join(model_dir, 'Model.h5'))
		except (ImportError, ValueError) as e:
			sys.exit('Error importing model.h5 file.' + os.path.join(model_dir, 'Model.h5') + 'No such file, or incompatible')

		with open(os.path.join(model_dir, 'Parameters.pickle'), 'rb') as parameters_file:
			parameters = pickle.load(parameters_file)
			self.results_dir        = parameters['results_dir']
			self.parameter_map      = parameters['parameter_map']
			self.image_types        = parameters['image_types']
			self.trained_parameters = parameters['trained_parameters']
			self.n_parameters       = len(self.trained_parameters)
			self.batch_size         = parameters['batch_size']


	def make_results_directory(self, run_id):
		results_dir = 'Run ' + str(run_id) + ' Results'
		try:
			os.makedirs(results_dir)
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				raise

		self.results_dir = results_dir


	def load_data(self, image_dir, image_types):
		print('Loading Data')
		image_names = os.listdir(image_dir)
		np.random.shuffle(image_names)

		# Remove the images types that aren't wanted for training if some are specified with a list
		if image_types is not None:
			image_names = [name for name in image_names if any(kind in name for kind in image_types)]

		n_names = len(image_names)
		print('n_images', n_names)

		x = np.array([Parameter_Support.get_image(os.path.join(image_dir, name)) for name in image_names])
		x = x.reshape((n_names, Parameter_Support.IMAGE_DIM, Parameter_Support.IMAGE_DIM, 1)).astype(np.float32)

		n_parameters = image_names[0].count('-') - 1
		y = np.ones(shape=(n_names, n_parameters), dtype=np.float32) * 50  # Initialize the array to the middle value in case some file names are missing data

		# Map the parameters in the parameter map dictionary
		temp_image_names = list(image_names)
		for name in temp_image_names:
			for letter in self.parameter_map:
				if '-' + letter in name:
					name = name.replace('-' + letter, '-' + self.parameter_map[letter])

		parameter_indexes = [m.start() + 1 for m in re.finditer('-', temp_image_names[0])][0:-1]  # Don't use the last '-' in the name. It's before the last number, not a parameter
		self.trained_parameters = [temp_image_names[0][i] for i in parameter_indexes]
		self.n_parameters = len(self.trained_parameters)
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

		test_split = int(x.shape[0] * 0.8)

		self.x_train, self.x_test = np.array_split(x, [test_split])
		self.y_train, self.y_test = np.array_split(y, [test_split])


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
			summary_file.write('Valid image types: ')
			for kind in image_types:
				summary_file.write(kind + ', ')

			summary_file.write('\n\n')

			self.model.summary(print_fn=lambda x: summary_file.write(x + '\n'))


	def save_model_and_params(self):
		self.model.save(os.path.join(self.results_dir, 'Model.h5'))

		to_save = {'results_dir': self.results_dir, 'parameter_map': self.parameter_map, 'image_types': self.image_types, 'trained_parameters': self.trained_parameters, 'batch_size': self.batch_size}
		with open(os.path.join(self.results_dir, 'Parameters.pickle'), 'wb') as parameter_file:
			pickle.dump(to_save, parameter_file)


	def train(self, n_epochs):
		history = self.model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=self.batch_size)

		self.train_predictions = self.model.predict(self.x_train, batch_size=self.batch_size)

		return history


	def evaluate_against_y(self):
		train_predictions_and_y = np.hstack((self.train_predictions, self.y_train))
		print('train predictions and y shape', train_predictions_and_y.shape)

		np.savetxt(os.path.join(self.results_dir, 'Train Results.csv'), train_predictions_and_y, delimiter=',', header=(','.join(self.trained_parameters) + ',') * 2)

		train_score = self.model.evaluate(self.x_train, self.y_train, batch_size=self.batch_size)
		print('Train Score: ', train_score)

		return train_score


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

		test_score = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)

		print('Test Score: ', test_score)

		return test_score


	def plot_predictions(self, predictions, title):
		train_figure, subplots = plt.subplots(1, self.n_parameters, figsize=(6 * self.n_parameters, 6))  # Create a subplot for each parameter

		for i in range(self.n_parameters):
			n_train = predictions.shape[0]

			predictions = predictions.reshape(n_train, 2)
			Parameter_Support.plot_results(subplots[i], predictions, self.trained_parameters[i])

	def plot_test_predictions_and_y(self, predictions, y, title, score):
		train_figure, subplots = plt.subplots(1, self.n_parameters, figsize=(6 * self.n_parameters, 6))  # Create a subplot for each parameter

		predictions_and_y = np.hstack((predictions, y))
		for i in range(self.n_parameters):
			n_train = predictions_and_y.shape[0]
			parameter_predictions_and_y = np.hstack((predictions_and_y[:, i].reshape(n_train, 1), predictions_and_y[:, self.n_parameters + i].reshape(n_train, 1)))
			parameter_predictions_and_y = parameter_predictions_and_y.reshape(n_train, 2)
			Parameter_Support.plot_results(subplots[i], parameter_predictions_and_y, score, self.trained_parameters[i])

		train_figure.canvas.set_window_title(title)
		train_figure.legend()

		train_figure.savefig(os.path.join(self.results_dir, title + '.png'), dpi=300)


	def set_parameter_map(self, new_parameter_map):
		# The parameter map is used equivocate one image parameter to another for training.
		# For example, to test how similar noise and roundness are, you could map f (noise) to r (roundness) with the map {'f':'r'}
		self.parameter_map = new_parameter_map


if __name__ == '__main__':
	main()


