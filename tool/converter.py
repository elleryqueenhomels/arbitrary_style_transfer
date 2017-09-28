# Extract pre-trained VGG19 weights from a matlab file,
# then save the weights into a npz file.

import numpy as np

from scipy.io import loadmat


VGG19_LAYERS = (
	'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

	'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

	'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
	'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

	'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
	'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

	'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
	'relu5_3', 'conv5_4', 'relu5_4'
)


def load_from_mat(weights_path):
	# extract the weights which are pre-trained on ImageNet dataset
	data = loadmat(weights_path)
	if not all(i in data for i in ('layers', 'classes', 'normalization')):
		raise ValueError('You are using the wrong VGG-19 data.')
	weights = data['layers'][0]
	return weights


def save_into_npz(weights, save_path):
	params = []
	count = 0
	for i, name in enumerate(VGG19_LAYERS):
		if name[:4] == 'conv':
			# matlabconv: [width, height, in_channels, out_channels]
			# tensorflow: [height, width, in_channels, out_channels]
			kernel, bias = weights[i][0][0][0][0]
			kernel = np.transpose(kernel, [1, 0, 2, 3])
			bias = bias.reshape(-1) # flatten
			params.append(kernel)
			params.append(bias)
			count += 2
	array = [count] + params
	np.savez(save_path, *array)
	print('count = %d' % count)


def load_from_npz(weights_path):
	weights = np.load(weights_path)
	count = int(weights['arr_0']) + 1
	params = []
	for i in range(1, count, 2):
		params.append(weights['arr_%s' % i])
		params.append(weights['arr_%s' % (i + 1)])
	return params


def check(weights, params):
	idx = 0
	for i, name in enumerate(VGG19_LAYERS):
		if name[:4] == 'conv':
			kernel, bias = weights[i][0][0][0][0]
			kernel = np.transpose(kernel, [1, 0, 2, 3])
			bias = bias.reshape(-1) # flatten
			if not np.all(kernel == params[idx]):
				print('Kernel is different! (i=%d, name=%s, idx=%d)' % (i, name, idx))
				return False
			if not np.all(bias == params[idx+1]):
				print('Bias is different! (i=%d, name=%s, idx=%d)' % (i, name, idx))
				return False
			idx += 2
	return True


# test
if __name__ == '__main__':
	from os.path import exists
	
	weights = load_from_mat('../pretrained/imagenet-vgg-verydeep-19.mat')

	save_path = '../pretrained/imagenet-vgg-19-weights.npz'

	if exists(save_path):
		print('\nThe npz file already exists!')
	else:
		save_into_npz(weights, save_path)
		print('\nSuccessfully save into npz file!')

	params = load_from_npz(save_path)

	print('\nNow begin to check the file...\n')
	
	result = check(weights, params)
	
	if result:
		print('All done! The file is correct!\n')
	else:
		print('Ooops! Something is wrong in the npz file!\n')

