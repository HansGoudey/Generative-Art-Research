# Generative-Art-Research
Neural networks to extract mid-level such as roundness, or messiness from images, especially abstract images. Training is done on a controlled set of generated images with continuous levels of these parameters.

- **Abstract-Feature-Recognition** contains the training, testing, and predicted code as well as helper files
- **Processing Sketches** Contains the Processing scripts used to generate the training images

## How to Train
Use **Parameters_Train.py**
### Get Images
#### Image Format
The images should be in the format **img-c##-r##-###.png**
The first three letters, **img** are the type of the image. Images generated by the same script should have the same type.
Each parameter **-c##** is designated by a dash followed by a letter. *c* might be *curliness.*
At the end is a string of numbers, **###**, used to keep multiple images with the same parameter values unique.

If you generate more images, remember to add the **type** to the **IMAGE_TYPES** variable in the train file.

    IMAGE_TYPES = ['mes', 'rdr', 'rns', 'rsc', 'img', 'rmr']

#### Download Link
The data used for the project during J-term is stored on basin, here:
http://www.cs.middlebury.edu/~hgoudey/Generative-Art-Research-Data/

The processing scripts in the repository could also just be run to generate the images again from scratch.

### Change Training Run Data
Change the location the data will be loaded from

    IMAGE_DIR = "E:\\Generative-Art-Research\\Images\\Messy Round Rects"
    
The number and letters of the parameters are gathered from the first image the script reads, so it assumes uniformity of the images in the IMAGE_DIR.

Change the output folder. The folder name is automatically created from these parameters:

    N_EPOCHS = 1
    RUN_ID = 302

The PARAMETER_MAP variable is used to make one parameter equivalent to another. Just add another element to the dictionary to collapse two parameters. 

### Run the Python File
The results are left in the previously mentioned folder.

## How to Predict
Use **Parameters_Predict.py**
### Remove Junk Images
Using a Firefox extension to download Google Images search results, I ended up with quite a few images that my script can't open. If the quality of the images is unknown, it would be worth it to run the **Remove_Bad_Images.py** script on the directory first.
### Change Prediction Run Data
Change the directory of the image data, and the directory that the model will be loaded from. The model h5 file and the Parameters.pickle file which stores the parameters used for training that model should be stored in the model directory.

    IMAGE_DIR = "E:\\Generative-Art-Research\\Images\\Messy Round Rects"
    
    MODEL_DIR = "E:\\Generative-Art-Research\\Abstract-Feature-Extraction\\Run 302 Results 1 Epochs"
### Run the Python File
The results of the training run will be stored in the same directory that the model comes from

## How to Test
The testing script is still being converted to use multiple parameters. The current script, **Parameters_Test.py** will only work with models created with the single-parameter training script, **Parameter_Train.py** (no "s").

# TO DO
- Create rounded rectangle images with no fill.
- Iterate on the network. Will smaller convolution filters from the start work as well? Explore more hyper-parameters.
- Symmetry Parameter
- Repetition Parameter
- Blur Parameter
- Try three parameters in one model.. four possibly?
- More... I will add to this list
