'''
-------------------------------------------------------------------------------
Testing out how to load and view the dataset
-------------------------------------------------------------------------------
Dataset retrieved from: https://github.com/hminle/gamut-mlp#dataset
Train Data: 2000 images
- we could use the zipped or unzipped folder
- the code below uses the unzipped folder
Test Data: 200 images
- split into 8 different zip files (I've just downloaded one of them for now)
'''

from PIL import Image

# let's take a look at the first image
img = Image.open('datasets/train/color_40.png')
img.show()

# TODO: create a DataLoader 
# TODO: retrieve the Clipped ProPhoto and Out-of-Gamut (OG) photos