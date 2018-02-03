import os
from PIL import Image

folder = "E:\Generative-Art-Research\Processing Sketches\Mess\Output"

for image_name in os.listdir(folder):
    try:
        with Image.open(image_name) as image:
            pass
    except IOError as e:
        print('Could not open', image_name, 'Deleting it')
        os.remove(os.path.join(folder, image_name))
