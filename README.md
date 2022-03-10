
# BITES: Balanced Individual Treatment Effect for Survival

**BITES** is a package for counterfactual survival analysis with the aim to predict the individual treatment effect of patients based on right-censored data.
It is using [PyTorch](https://pytorch.org), and main functioality of [pycox](https://github.com/havakv/pycox).
To balance generating distributions of treatment and control group it calculates the Sinkhorn divergence using [geomloss](https://www.kernel-operations.io/geomloss/).
Additionally, it is set up for automatic hyper-parameter optimization using [ray[tune]](https://docs.ray.io/en/latest/tune/index.html).

The package includes an easy to use framework for [BITES](https://arxiv.org/abs/2201.03448) and [DeepSurv](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1) both as single model and T-learner.
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

### The config file
The complete workflow for (B)ITES, (T-)DeepSurv and CFRNet is controllable by setting the ``config`` parameters
````python
config = {
    "Method": 'BITES',                  # or 'ITES', 'DeepSurvT', 'DeepSurv', 'CFRNet'
    "trial_name": 'Simulation3',        # name of your trial
    "result_dir": './ray_results',      # directory for the results
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
````
Both, the [Raytune](https://docs.ray.io/en/latest/tune/index.html) search routines and fixed values can be used for the hyper-parameter optimization.


### Examples
We include two example scripts for both Simulated and RGBSG data[[4,5]](#4) as discussed in [BITES](https://arxiv.org/abs/2201.03448) [[1]](#1).
To train Bites on one of the Simulated datasets run [Simulation_run.py](https://github.com/sschrod/BITES/blob/main/examples/Simulation_run.py).
The default is set to the non-linear Simulation with treatment bias, with a single set of hyper-parameters. 
The results can be analysed with [Simulation_analyse.py](https://github.com/sschrod/BITES/blob/main/examples/Simulation_analyse.py).

To train BITES on the RGBSG data you need to dowload the [dataset](https://github.com/arturomoncadatorres/deepsurvk/tree/master/deepsurvk/datasets/data) and add `rgbsg.h5` to ``examples/data/RGBSG``
We include an example model that can be loaded with [RGBSG_analyse.py](/BITES/examples/RGBSG_analyse.py). To do your own analysis use [RGBSG_run.py](https://github.com/sschrod/BITES/blob/main/examples/RGBSG_run.py).


If you are using Docker run to start the bites docker and mount your current Working directory into the bites Docker.
````sh
    docker run --gpus all -it --rm -v $PWD:/mnt bites python3 /mnt/RGBSG_run.py
    docker run --gpus all -it --rm  -v $PWD:/mnt bites python3 /mnt/RGBSG_analyse.py
````


## Use your own Data
To use BITES for your own data simply call the function
````python
from bites.model.Fit import fit
fit(config, X_train, Y_train, event_train, treatment_train)
````
To load the best model (according to validation loss) use
````python
from bites.analyse.analyse_utils import get_best_model
model=get_best_model(config)
````
For further anaysis you can use
````python
from bites.analyse.analyse_utils import analyse
analyse(config, X_train,Y_train,event_train,treatment_train,X_test,Y_test,event_test,treatment_test)
````

## Additional Features
### CFRNet
Using CFRNet will ignore the event indicator and assume complete, non-censored outcomes.

### DeepSurv without Treatment assignemnt
DeepSurv can be used without Treatment assignment[[6]](#6). Just set ```treatment_train=None``` to only consider a single survival model.


## References

[1] Stefan Schrod, et. al. BITES: Balanced Individual Treatment Effect for Survival data, 2022. [[arXiv]](https://arxiv.org/abs/2201.03448).

[2] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks and Cox regression. *Journal of Machine Learning Research*, 20(129):1–30, 2019. [[paper]](http://jmlr.org/papers/v20/18-424.html).

[3] Uri Shalit, Fredrik D. Johansson, and David Sontag. Estimating individual treatment effect: generalization bounds and algorithms, 2016. [[arXiv]](http://arxiv.org/pdf/1606.03976v5).

[4] J. A. Foekens, et al., The urokinase system of plasminogen activation and prognosis in 2780 breast cancer patients. *Cancer research*, 60(3):636–643, 2000. [[paper]](https://pubmed.ncbi.nlm.nih.gov/10676647/).


[5] Claudia Schmoor, et al., Randomized and non-randomized patients in clinical trials: Experiences with comprehensive cohort studies. *Statistics in Medicine*, 15(3):263–271, 1996. [[paper]](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0258(19960215)15:3%3C263::AID-SIM165%3E3.0.CO;2-K)

[6] Jared Katzman,et al., DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network. *BMC Medical Research Methodology*, 18(1):1, 2018. [[arXiv]](http://arxiv.org/pdf/1606.00931v3).