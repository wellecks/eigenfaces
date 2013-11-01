# for image reading / writing
from scipy.misc import *
from scipy import linalg
import glob
import numpy
import os

in_dir  = "bw_img"
out_dir = "output"
img_dims = (200, 180)

# returns a n x p array (n images with p pixels each)
def load_images(directory):
	# get a list of all the picture filenames
	jpgs = glob.glob(directory + '/*.jpg')
	# load a greyscale version of each image
	imgs = numpy.array([imread(i, True).flatten() for i in jpgs])
	return imgs

# choose one image from each folder
def choose_images(indir, outdir):
	counter = 0
	for folder in glob.glob(indir + '/*'):
		filenames = glob.glob(folder + '/*.jpg')
		# just choose the first image
		img = imread(filenames[0], True)
		imsave(outdir + "/img_" + str(counter) + ".jpg", img)
		counter = counter + 1

def pca(data):
	mu = numpy.mean(data, 0)

	# save mean photo
	imsave(out_dir + "/mean.jpg", mu.reshape(img_dims))

	# mean adjusted data
	ma_data = data - mu

	# run SVD
	e_faces, sigma, v = linalg.svd(ma_data.transpose(), full_matrices=False)

	# save eigenfaces
	for i in range(e_faces.shape[1]):
		save_image("eigenfaces", i, e_faces[:,i])

	# compute weights for each image
	weights = numpy.dot(ma_data, e_faces)

	return e_faces, weights, mu

def reconstruct(img_idx, e_faces, weights, mu, npcs):
	# reconstruct by dotting weights with the eigenfaces and adding to mean
	img = mu + numpy.dot(weights[img_idx, 0:npcs], e_faces[:, 0:npcs].T)
	img_id = (npcs / 10) * "a" + str(npcs % 10)
	save_image("reconstructed/" + str(img_idx), img_id, img)
	
def save_image(subdir, img_id, data):
	directory = out_dir + "/" + subdir
	if not os.path.exists(directory): os.makedirs(directory)
	imsave(directory + "/image_" + str(img_id) + ".jpg", data.reshape(img_dims))

def run_all_reconstructions():
	data = load_images(in_dir)
	e_faces, weights, mu = pca(data)
	for p in range(data.shape[0]):
		for i in range(data.shape[0]):
			reconstruct(p, e_faces, weights, mu, i)

