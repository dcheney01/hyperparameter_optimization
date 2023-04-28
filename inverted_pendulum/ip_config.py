from ray import tune

ip_config = {
        'project_name': 'hyperparam_opt_ip',
        'run_name': 'data-points-runtest',

        'n_inputs': 3, # [theta, theta_dot, tau] at t
        'n_outputs': 2, #[theta, theta_dot] at t+1

        # Parameters to Optimize
        'b_size': tune.choice([16, 32, 64, 128, 256, 512]),
        'n_hlay': tune.choice(list(i for i in range(11))),
        'hdim': tune.choice(2**i for i in range(3, 11)),
        'lr': tune.loguniform(1e-5, 1e-1),
        'act_fn': tune.choice(['relu', 'tanh']),#, 'sigmoid']),#, 'softmax']),
        'loss_fn': tune.choice(['mse', 'mae']),#, 'cross_entropy']),
        'opt': tune.choice(['adam', 'sgd']),# 'ada', 'lbfgs', 'rmsprop']),
        'nn_arch': 'simple_fnn',

        # Other Configuration Parameters
        'accuracy_tolerance': 0.01, # This translates to about 1/2 a degree for inverted pendulum
        'calculates_xdot':False,
        'num_workers': 6,
        'generate_new_data': True,
        'learn_mode': 'x',
        'dataset_size': 10000,
        'normalized_data': False,
        'dt': 0.01,
        'cpu_num': 7,
        'gpu_num': 0.8,
        
        # Optimization Tool Parameters
        'max_epochs': 2,
        'num_samples': 2,
        'path': '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/',
        }


test_ip_config = {
        'project_name': 'test',
        'run_name': 'test',

        'n_inputs': 3, # [theta, theta_dot, tau] at t
        'n_outputs': 2, #[theta, theta_dot] at t+1

        # Parameters to Optimize
        'b_size': 32,
        'n_hlay': 3,
        'hdim': 20,
        'lr': 0.001,
        'act_fn': 'relu',
        'loss_fn': 'mse',
        'opt': 'adam',
        'nn_arch': 'simple_fnn',

        # Other Configuration Parameters
        'accuracy_tolerance': 0.01, # This translates to about 1/2 a degree
        'calculates_xdot':False,
        'num_workers': 6,
        'generate_new_data': False,
        'learn_mode': 'x',
        'dataset_size': 1000,
        'normalized_data': False,
        'dt': 0.01,
        'cpu_num': 5,
        'gpu_num': 0.65,

        # Optimization Tool Parameters
        'max_epochs': 2,
        'num_samples': 1,
        'path': '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/data/test/',
        }