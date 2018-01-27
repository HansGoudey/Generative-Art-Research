import os

folder = "E:\\Generative-Art-Research\\Images\\Noise"

for name in os.listdir(folder):
    os.rename(os.path.join(folder, name), os.path.join(folder, 'rns' + name[3:]))




