from scipy.misc import *
from scipy import linalg
import glob
import numpy
import os
import pdb

# loads a greyscale version of every jpg image in the directory.
# INPUT  : directory
# OUTPUT : imgs - n x p array (n images with p pixels each)
def load_images(directory):
	# get a list of all the picture filenames
	jpgs = glob.glob(directory + '/*.jpg')
	# load a greyscale version of each image
	imgs = numpy.array([imread(i, True).flatten() for i in jpgs])
	return imgs

# chooses the first image from each folder in indir,
# and saves a copy of the images in the outdir.
# INPUT  : indir  - directory to retrieve folders from
#          outdir - directory to save images to
def choose_images(indir, outdir):
	counter = 0
	for folder in glob.glob(indir + '/*'):
		filenames = glob.glob(folder + '/*.jpg')
		# just choose the first image
		img = imread(filenames[0], True)
		imsave(outdir + "/img_" + str(counter) + ".jpg", img)
		counter = counter + 1

# Run Principal Component Analysis on the input data.
# INPUT  : data    - an n x p matrix
# OUTPUT : e_faces -
#          weights -
#          mu      -
def pca(data):
	mu = numpy.mean(data, 0)
	# mean adjust the data
	ma_data = data - mu
	# run SVD
	e_faces, sigma, v = linalg.svd(ma_data.transpose(), full_matrices=False)
	pdb.set_trace()
	# compute weights for each image
	weights = numpy.dot(ma_data, e_faces)
	return e_faces, weights, mu

# reconstruct an image using the given number of principal
# components.
def reconstruct(img_idx, e_faces, weights, mu, npcs):
	# dot weights with the eigenfaces and add to mean
	recon = mu + numpy.dot(weights[img_idx, 0:npcs], e_faces[:, 0:npcs].T)
	return recon
	
def save_image(out_dir, subdir, img_id, img_dims, data):
	directory = out_dir + "/" + subdir
	if not os.path.exists(directory): os.makedirs(directory)
	imsave(directory + "/image_" + str(img_id) + ".jpg", data.reshape(img_dims))

def run_experiment():
	in_dir  = "bw_img"
	out_dir = "output"
	img_dims = (200, 180)

	data = load_images(in_dir)
	e_faces, weights, mu = pca(data)

	# save mean photo
	#imsave(out_dir + "/mean.jpg", mu.reshape(img_dims))
	
	# save each eigenface as an image
	for i in range(e_faces.shape[1]):
		continue
		#save_image(out_dir, "eigenfaces", i, e_faces[:,i])

	# reconstruct each face image using an increasing number
	# of principal components	
	reconstructed = []
	for p in range(data.shape[0]):
		for i in range(data.shape[0]):
			reconstructed.append(reconstruct(p, e_faces, weights, mu, i))
			img_id = (i / 10) * "a" + str(i % 10)
			#save_image(out_dir, "reconstructed/" + str(p), img_id, img)
