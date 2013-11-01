from images2gif import writeGif
from PIL import Image
import os
import glob

for person in range(0, 113):
	folder = "output/reconstructed/" + str(person)
	file_names = glob.glob(folder + '/*.jpg')

	images = [Image.open(fn) for fn in file_names]

	filename = "transform_" + str(person) + ".gif"
	writeGif(filename, images, duration=0.15)