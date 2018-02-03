import os

folder = "E:\Generative-Art-Research\Processing Sketches\Mess\Output"

for name in os.listdir(folder):
    # os.rename(os.path.join(folder, name), os.path.join(folder, 'rns' + name[3:]))
    # os.rename(os.path.join(folder, name), os.path.join(folder, name[:-4] + 'b' + name[-4:]))

    if 'f' in name:
        os.rename(os.path.join(folder, name), os.path.join(folder, 'rns' + name[3:]))





