
# Generative-Art-Research
Neural networks to extract mid-level such as roundness, or messiness from images, especially abstract images. Training is done on a controlled set of generated images with continuous levels of these parameters.

- **Abstract-Feature-Recognition** contains the training, testing, and predicted code as well as helper files
- **Processing Sketches** Contains the Processing scripts used to generate the training images

## How to Train on an Image Set
Run the python file Parameters.py like this:

    python3 Parameters.py train IMAGE_DIR -l MODEL_TO_LOAD -n RUN_ID -e EPOCHS -t IMAGE_TYPES -m PARAMETER_MAP

 - **IMAGE_DIR (required):** The directory to load the images from
 - **MODEL_TO_LOAD (-l):** If this field is specified, the program will look in the folder provided and start the training with the model loaded from there.
 - **RUN_ID (-n):** The results will be saved in a folder with this name. If this is not specified, a name will be created based on other information.
 - **EPOCHS (-e):** The number of epochs to train on the data
 - **IMAGE_TYPES (-t):** Types of images to train on
 - **PARAMETER_MAP (-t):** Mapping of parameters used to equate them for training. A dictionary of one letter strings.


## How to Predict
Run the python file Parameters.py like this:

    python3 Parameters.py predict IMAGE_DIR -l MODEL_TO_LOAD -n RUN_ID
    
 - **IMAGE_DIR (required):** The directory to load the images from
 - **MODEL_TO_LOAD (required)(-l):** Model used to predict.
 - **RUN_ID (-n):** The results will be saved in a folder with this name. If this is not specified, the loaded model's folder will be used
 
 Other options aren't included for now for predicting, because prediction images are assumed to be unlabeled.
 
## How to Test
Run the python file Parameters.py like this:

    python3 Parameters.py test IMAGE_DIR -l MODEL_TO_LOAD -n RUN_ID -e EPOCHS -t IMAGE_TYPES -m PARAMETER_MAP
    
 - **IMAGE_DIR (required):** The directory to load the images from
 - **MODEL_TO_LOAD (required)(-l):** Model used to test
 - **RUN_ID (-n):** Results are saved in a folder with this name if the field is specified, otherwise it uses the existing folder.
 - **IMAGE_TYPES (-t):** Types of images to test on.
 - **PARAMETER_MAP (-t):** Mapping of parameters used to equate them for testing. A dictionary of one letter strings.


## Image Format and Files
The images should be in the format **img-c##-r##-###.png**
The first three letters, **img** are the type of the image. Images generated by the same script should have the same type.
Each parameter **-c##** is designated by a dash followed by a letter. *c* might be *curliness.*
At the end is a string of numbers, **###**, used to keep multiple images with the same parameter values unique.

The data used for the project during J-term is stored on basin, here:
http://www.cs.middlebury.edu/~hgoudey/Generative-Art-Research-Data/

The processing scripts in the repository could also just be run to generate the images again from scratch.
### Remove Junk Images
Using a Firefox extension to download Google Images search results, I ended up with quite a few images that my script can't open. If the quality of the images is unknown, you can run the **Remove_Bad_Images.py** script on the directory first.



# TO DO
- Iterate on the network. Will smaller convolution filters from the start work as well? Explore more hyper-parameters.
- Symmetry Parameter
- Repetition Parameter
- Blur Parameter
- Try three parameters in one model.. four possibly?
- More... I will add to this list
