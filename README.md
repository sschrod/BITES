
# BITES: Balanced Individual Treatment Effect for Survival

**BITES** is a package for counterfactual survival analysis with the aim to predict the individual treatment effect of patients based on right-censored data.
It is using [PyTorch](https://pytorch.org), main functioality of [pycox](https://github.com/havakv/pycox)
and clauclates the Sinkhorn divergence using [geomloss](https://www.kernel-operations.io/geomloss/).
Additionally, it is set up for automatic hyper-parameter optimization using [ray[tune]](https://docs.ray.io/en/latest/tune/index.html).

The package includes an easy to use framework for [BITES](TODO) and [DeepSurv](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1) as both single model and T-learner.
Additionally, for non-censored data it includes the Counterfactual Regression Network [CFRNet](https://arxiv.org/pdf/1606.03976.pdf) [[1]](#3).

## Get started
We recommend setting up [PyTorch](https://pytorch.org) with cuda support if you have GPUs available.
The package is tested with torch==1.9.1+cu111 with the most recent CUDA 11.4. 

The easiest way to use BITES with all supported features is to build the provided Dockerfile
```shell
docker build -t bites -f Dockerfile_BITES .
```

Local installation is possible using pip install
```sh
  cd BITES
  pip install bites
  pip install -r requirements.txt
  
```



## Usage
### Example
We include two example scripts based on the discussed cases in our BITES-Paper(ADD LINK).
For the simulated case run the...
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
````latex {cmd=true hide=true}
\documentclass{standalone}
\usepackage{tikz}
\usetikzlibrary{matrix}
\begin{document}
\begin{tikzpicture}
  \matrix (m) [matrix of math nodes,row sep=3em,column sep=4em,minimum width=2em]
  {
     F & B \\
      & A \\};
  \path[-stealth]
    (m-1-1) edge node [above] {$\beta$} (m-1-2)
    (m-1-2) edge node [right] {$\rho$} (m-2-2)
    (m-1-1) edge node [left] {$\alpha$} (m-2-2);
\end{tikzpicture}
\end{document}
````







## Use your own Data
To use BITES for your own data simply call the function
````python
from BITES import run
run(config,X,y,event,treatment)
````
And to analyse the results use
````python
from BITES import analyse
analyse(config, X_train,Y_train,event_train,treatment_train,X_test,Y_test,event_test,treatment_test)
````
This will load the best trial with respect to the validation loss and return the achived C-Index on the test set.
To load the model directly call
````python
from Bites.bites import get_best_model
model=get_best_model(config)
````
[some_file.py](link) 


##References

[1] Stefan Schrod, Andreas Schäfer, and Michael altenbuchinger. BITES:... . *Some Journal*, , 2022. [[paper](link)]

[2] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks and Cox regression. *Journal of Machine Learning Research*, 20(129):1–30, 2019. [[paper](http://jmlr.org/papers/v20/18-424.html)]

[3] Shalit
