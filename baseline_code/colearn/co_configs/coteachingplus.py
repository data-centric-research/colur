config = {
    'algorithm' : 'Coteachingplus',
    # dataset param
    'dataset' : 'cifar-10',
    'input_channel' : 3,
    'num_classes' : 10,
    'root' : './data',
    'noise_type' : 'sym',
    'percent' : 0.2,
    'seed' : 7895,
    # model param
    'model1_type' : 'resnet18',
    'model2_type' : 'resnet18',
    # train param
    'batch_size' : 64,
    'lr' : 0.001,
    'epochs' : 11,
    'num_workers' : 4,
    'exponent' : 1,
    'adjust_lr' : 1,
    'num_gradual' : 1,
    'forget_rate' : 0.8,
    'epoch_decay_start' : 5,
    # result param
    'save_result' : True
    }