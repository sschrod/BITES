from bites.data.RGBSG.RGBSG_utilis import load_RGBSG
from ray import tune
from bites.model.Fit import fit

if __name__ == '__main__':
    """ config for ITES
    config = {
        "Method": 'ITES',
        "trial_name": 'RGBSG',
        "result_dir": './ray_results',
        "val_set_fraction": 0.2,
        "num_covariates": 9,
        "shared_layer": tune.grid_search([[7,5]]),
        "individual_layer": tune.grid_search([[5,3],[3]]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "dropout": tune.choice([0.1,0.2]),
        "weight_decay": tune.choice([0.01,0.1]),
        "batch_size": 3000,
        "epochs": 5000,
        "alpha": 0.0,
        "blur": 0.05,
        "grace_period": 50,
        "gpus_per_trial": 0.25,
        "cpus_per_trial": 2,
        "num_samples": 10,
        "pin_memory": True
    }
    """

    config = {
        "Method": 'BITES',
        "trial_name": 'RGBSG_v3',
        "result_dir": './ray_results',
        "val_set_fraction": 0.2,
        "num_covariates": 9,
        "shared_layer": tune.grid_search([[7,5]]),
        "individual_layer": tune.grid_search([[5,3],[3]]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "dropout": tune.choice([0.1,0.2]),
        "weight_decay": tune.choice([0.01,0.1]),
        "batch_size": 3000,
        "epochs": 5000,
        "alpha": tune.grid_search([0.001,0.01,0.1,1,10]),
        "blur": tune.grid_search([0.05,0.1]),
        "grace_period": 50,
        "gpus_per_trial": 0.25,
        "cpus_per_trial": 2,
        "num_samples": 100,
        "pin_memory": True
    }


    """
    config = {
        "Method": 'DeepSurvT',
        "trial_name": 'RGBSG',
        "result_dir": './ray_results',
        "val_set_fraction": 0.2,
        "num_covariates": 9,
        "shared_layer": tune.grid_search([[7,5]]),
        "individual_layer": tune.grid_search([[5,3],[3]]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "dropout": tune.choice([0.1,0.2]),
        "weight_decay": tune.choice([0.01,0.1]),
        "batch_size": 3000,
        "epochs": 5000,
        "alpha": 0.0,
        "blur": 0.05,
        "grace_period": 50,
        "gpus_per_trial": 0.25,
        "cpus_per_trial": 2,
        "num_samples": 25,
        "pin_memory": True
    }
    """

    X_train, Y_train, event_train, treatment_train, _, _ = load_RGBSG(partition='train',
                                                                      filename_="./data/RGBSG/rgbsg.h5")
    #TODO: write function that checks input data and config
    if config["num_covariates"]!=X_train.shape[1]:
        print('config[num_covariates] has to match the shape of the training data')
        print('Resetting config[num_covariates]')
        config["num_covariates"]=X_train.shape[1]


    fit(config, X_train=X_train, Y_train=Y_train, event_train=event_train, treatment_train=treatment_train)


