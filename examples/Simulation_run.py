from ray import tune
from bites.model.Fit import fit
import pickle

if __name__ == '__main__':

    """Simple config for single set of hyperparameters with suggestions for setting up the tune hyper-parameter search"""
    config = {
        "Method": 'DeepSurv',                  # or 'ITES', 'DeepSurvT', 'DeepSurv', 'CFRNet'
        "trial_name": 'Simulation3',        # name of your trial
        "result_dir": './ray_results',      # will be created
        "val_set_fraction": 0.2,            # Size of the validation Set
        "num_covariates": 20,               # Number of covariates in the data
        "shared_layer": [15, 10],           # or just tune.grid_search([<list of lists>])
        "individual_layer": [10, 5],        # or just tune.grid_search([<list of lists>])
        "lr": tune.loguniform(1e-4, 1e-1),  # or fixed value,e.g. 0.001
        "dropout": 0.1,                     # or tune.choice([<list values>])
        "weight_decay": 0.2,                # or tune.choice([<list values>])
        "batch_size": 3000,                 # or tune.choice([<list values>])
        "epochs": 10000,
        "alpha": 0.1,                       # or tune.grid_search([<list values>])
        "blur": 0.05,                       # or tune.grid_search([<list values>]),
        "grace_period": 50,                 # Early stopping
        "gpus_per_trial": 0,                # For GPU support set >0 (fractions of GPUs are supported)
        "cpus_per_trial": 16,               # scale according to your resources
        "num_samples": 1,                   # Number the run is repeated
        "pin_memory": True                  # If the whole data fits on the GPU memory, pin the memory to speed up computation
    }

    num_Samples = 0       # int 0 to 4 for 1000 to 4000 training samples
    X_train, y_train = pickle.load(open('data/Simulation_Treatment_Bias/train_data.Sim3', 'rb'))[num_Samples]

    """Fit the model"""
    fit(config, X_train=X_train, Y_train=y_train[:,5], event_train=y_train[:,4], treatment_train=y_train[:,2])

