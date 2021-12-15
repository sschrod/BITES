from bites.data.RGBSG.RGBSG_utilis import load_RGBSG
from bites.model.Fit import fit

if __name__ == '__main__':
    X_train, Y_train, event_train, treatment_train,_,_ = load_RGBSG(partition='train', filename_="../bites/data/RGBSG/rgbsg.h5")
    X_test, Y_test, event_test,treatment_test,_,_ = load_RGBSG(partition='test', filename_="../bites/data/RGBSG/rgbsg.h5")

    config = {
        "Method": 'BITES',
        "trial_name": 'RGBSG',
        "result_dir": './ray_results',
        "val_set_fraction": 0.2,
        "num_covariates": 20,
        "shared_layer": [15, 10],
        "individual_layer": [5],
        "lr": 0.01,
        "dropout": 0.1,
        "weight_decay": 0.1,
        "batch_size": 3000,
        "epochs": 10000,
        "alpha": 0.0,
        "blur": 0.05,
        "grace_period": 20,
        "gpus_per_trial": 1,
        "cpus_per_trial": 4,
        "num_samples": 4,
        "pin_memory": True
    }


    if config["num_covariates"]!=X_train.shape[1]:
        print('config[num_covariates] has to match the shape of the training data')
        print('Resetting config[num_covariates]')
        config["num_covariates"]=X_train.shape[1]




    fit(config, X_train=X_train, Y_train=Y_train, event_train=event_train, treatment_train=treatment_train)


