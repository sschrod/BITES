
# BITES: Balanced Individual Treatment Effect for Survival

**BITES** is a package for counterfactual survival analysis with the aim to predict the individual treatment effect of patients based on right-censored data.
It is using [PyTorch](https://pytorch.org), and main functioality of [pycox](https://github.com/havakv/pycox).
To balance generating distributions of treatment and control group it calculates the Sinkhorn divergence using [geomloss](https://www.kernel-operations.io/geomloss/).
Additionally, it is set up for automatic hyper-parameter optimization using [ray[tune]](https://docs.ray.io/en/latest/tune/index.html).

The package includes an easy to use framework for [BITES](TODO)(AddLink) and [DeepSurv](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1) both as single model and T-learner.
Additionally, to analyse non-censored data it includes the Counterfactual Regression Network [CFRNet](https://arxiv.org/pdf/1606.03976.pdf) [[3]](#3).

## Get started
We recommend setting up [PyTorch](https://pytorch.org) with cuda if you have GPUs available.
The package is tested with torch==1.9.1+cu111 working with most recent CUDA 11.4. 


To install the package from source clone the directory and use pip
```sh
  git clone https://github.com/sschrod/BITES.git
  cd BITES
  pip install .
  pip install -r requirements.txt
  
```

Alternatively, you can build a Docker image with
```shell
docker build -t bites -f Dockerfile_BITES .
```





## Usage
### Example
We include two example scripts for both Simulated and application on the RGBSG data as discussed in our [paper](ADD LINK)[[1]](#1).
To train Bites on one of the Simulated datasets run 

continue here!!!!!!!!!!!!!!!!!!!!!!

To train BITES on the RGBSG data run [RGBSG_run.py](/BITES/examples/RGBSG_run.py). This starts hyper-parameter optimization for the network settings specified in `config` (see below).
Results are saved in `/<result_dir>/<trial_name>` and can be analised by [RGBSG_analyse.py](/BITES/examples/RGBSG_analyse.py).

If you are using Docker run to start the bites docker and mount your current Working directory into the bites Docker.
````sh
    docker run --gpus all -it --rm --shm-size=100gb -v $PWD:/mnt bites python3 /mnt/RGBSG_run.py
````
and analyse the findings with
````sh
    docker run --gpus all -it --rm --shm-size=100gb -v $PWD:/mnt bites python3 /mnt/RGBSG_analyse.py
````

### The config file
The complete workflow for BITES, DeepSurv and CFRNet are completely controllable by setting the ``config`` parameters
````python
config = {
    "Method": 'bites', #'ITES', 'DeepSurv', 'CFRNet'
    "trial_name": 'RGBSG',
    "result_dir": './ray_results',
    "val_set_fraction": 0.2,
    "num_covariates": 9,
    "shared_layer": tune.grid_search([[7,5]]),  # or just [7,5]
    "individual_layer": tune.grid_search([[5,3],[3]]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "dropout": tune.choice([0.1,0.2]),
    "weight_decay": tune.choice([0.01,0.1]),
    "batch_size": 3000,
    "epochs": 5000,
    "alpha": tune.grid_search([0.001,0.01,0.1,1,10]),
    "blur": tune.grid_search([0.05,0.1]),
    "grace_period": 50,
    "gpus_per_trial": 0.25, # Set to 0 for CPU only
    "cpus_per_trial": 2,
    "num_samples": 10,
    "pin_memory": True
    }
````
Here we use the search routines provided by [Raytune](https://docs.ray.io/en/latest/tune/index.html). However, also single values can be passed instead.

### The network Architecture
The BITES architecture is given by








## Use your own Data
To use BITES for your own data simply call the function
````python
from bites.model.Fit import fit
fit(config, X_train, Y_train, event_train, treatment_train)
````
And to analyse the results use
````python
from bites.analyse.analyse_utils import analyse
analyse(config, X_train,Y_train,event_train,treatment_train,X_test,Y_test,event_test,treatment_test)
````
This will load the best trial with respect to the validation loss and return the achived C-Index on the test set.
To load the model directly call
````python
from bites.analyse.analyse_utils import get_best_model
model=get_best_model(config)
````
[some_file.py](link) 


##References

[1] Stefan Schrod, Andreas Schäfer, and Michael altenbuchinger. BITES:... . *Some Journal*, , 2022. [[paper](link)]

[2] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks and Cox regression. *Journal of Machine Learning Research*, 20(129):1–30, 2019. [[paper](http://jmlr.org/papers/v20/18-424.html)]

[3] Shalit
