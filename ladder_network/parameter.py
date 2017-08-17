
my_parameter = {
    'batch':100,
    'epochs':100,
    'learning_rate':0.02,
    'noise_std':0.3,
    'data_dir':'../data',
    'seed':42,
    'unsupervised_cost_lambda': [0.1, 0.1, 0.1, 0.1, 0.1, 10., 20000.],
    'cuda':True,
    'decay_epoch':20,
    'encoder_activations':['relu', 'relu', 'relu', 'relu', 'relu', 'softmax'],
    'encoder_sizes': [1000, 800, 600, 400, 200, 10],
    'decoder_sizes': [200, 400, 600, 800, 1000, 784],
    'encoder_train_bn_scaling': [True, True, True, True, True, True]
}
