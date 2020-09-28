network_layers = model.layers
feature_layers = ['block6a_expand_activation', 'block4a_expand_activation','block3a_expand_activation', 'block2a_expand_activation']
with open('network.txt', 'w+') as f:
	for layer in network_layers:
		f.write('{}: {}\n'.format(layer.name, layer.output.get_shape()))
		if layer.name in feature_layers:
			f.write('\nFeature extansion ---{}\n'.format(layer.name))
