from ray import tune

ip_config = {
        'project_name': 'hyperparam_opt_ip',
        'run_name': 'fnn_optimization_try3',

        'nn_arch': 'simple_fnn', # 'lstm', 'transformer'

        # Parameters to Optimize
        'b_size': tune.choice([16, 32, 64, 128, 256, 512]),
        'n_hlay': tune.choice(list(i for i in range(3))),
        'hdim': tune.choice(list(2**i for i in range(6, 11))),
        'lr': tune.loguniform(1e-5, 1e-2),
        'act_fn': tune.choice(['relu']),#, 'tanh', 'sigmoid']),
        'loss_fn': tune.choice(['mse']),# 'mae']),#, 'cross_entropy']),
        'opt': tune.choice(['adam']), #'sgd']),# 'ada', 'lbfgs', 'rmsprop']),

        # For Transformer
        # 'b_size': tune.choice([16, 32, 64, 128, 256, 512]),
        # 'n_hlay': tune.choice(list(i for i in range(11))),
        # 'hdim': tune.choice(list(2**i for i in range(3, 11))),
        # 'lr': tune.loguniform(1e-5, 1e-2),
        # 'act_fn': tune.choice(['relu']),#, 'tanh', 'sigmoid']),
        # 'loss_fn': tune.choice(['mse', 'mae']),#, 'cross_entropy']),
        # 'opt': tune.choice(['adam', 'sgd']),# 'ada', 'lbfgs', 'rmsprop']),


        'n_inputs': 3, # [theta, theta_dot, tau] at t
        'n_outputs': 2, #[theta, theta_dot] at t+1

        # Other Configuration Parameters
        'accuracy_tolerance': 0.01, # This translates to about 1/2 a degree for inverted pendulum
        'num_workers': 6,
        'generate_new_data': False,
        'learn_mode': 'x',
        'dataset_size': 60000,
        'normalized_data': True,
        'dt': 0.01,
        'cpu_num': 7,
        'gpu_num': 0.8,
        
        # Optimization Tool Parameters
        'max_epochs': 750,
        'num_samples': 200,
        'path': '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/',
        }


test_ip_config = {
        'project_name': 'test',
        'run_name': 'test',

        'nn_arch': 'simple_fnn',

        # Parameters to Optimize
        'b_size': 16,
        'n_hlay': 0,
        'hdim': 512,
        'lr': 0.0001,
        'act_fn': 'relu',
        'loss_fn': 'mse',
        'opt': 'adam',

        'n_inputs': 3, # [theta, theta_dot, tau] at t
        'n_outputs': 2, #[theta, theta_dot] at t+1

        # Other Configuration Parameters
        'accuracy_tolerance': 0.01, # This translates to about 1/2 a degree
        'num_workers': 6,
        'generate_new_data': True,
        'learn_mode': 'x',
        'dataset_size': 60000,
        'normalized_data': False,
        'dt': 0.01,
        # 'cpu_num': 7,
        # 'gpu_num': 0.9,

        # Optimization Tool Parameters
        'max_epochs': 500,
        # 'num_samples': 1,
        'path': '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/data/test/',
        }