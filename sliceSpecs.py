import os
from PIL import Image
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--raw_specs', default='data/raw_specs', help="Directory with the raw spectrograms")
parser.add_argument('--img_size', default = 128, help="spectrogram slice image size")
parser.add_argument('--output_dir', default='data/sliced_specs', help="Directory with the spectrogram slices")



def slice_spec(fname, size, input_dir, output_dir):
	im = Image.open(os.path.join(input_dir, fname))
	w,h = im.size
	num_slices = int(w/size)

	for i in range(num_slices):
		tmp_im = im.crop((i*size, 0, i*size+size, size)); #create a temporary image
		tmp_im.save(os.path.join(output_dir, fname.split(".")[0]+"{}.png".format(i))) # save in output




if __name__ == '__main__':
	args = parser.parse_args()

	raw_specs = args.raw_specs
	sliced_specs = args.output_dir
	size = args.img_size;

	assert os.path.isdir(raw_specs), "Couldn't find the dataset at {}".format(raw_specs)

	if not os.path.exists(sliced_specs):
		os.mkdir(sliced_specs)
	else:
		print("Warning: output dir {} already exists".format(sliced_specs))

	for fname in tqdm(os.listdir(raw_specs)):
		if fname.endswith(".png"):
			slice_spec(fname, size, raw_specs, sliced_specs)

