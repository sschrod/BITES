
import os
from functools import partial

import numpy as np
import ray
import torch
from bites.model.BITES_base import BITES, BITES_Loss
from bites.model.CFRNet_base import CFRNet_Loss
from bites.model.DeepSurv_base import DeepSurv, DeepSurv_Loss
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import *


def fit (config, X_train, Y_train, event_train=None, treatment_train=None,**kwargs):
    """
    :param config: config file as given in the examples.
    :param X_train: np.array(num_samples, features)
    :param Y_train: np.array(num_samples,)
    :param event_train: np.array(num_samples,)
    :param treatment_train: np.array(num_samples,)
    :param kwargs:
    :return: tune.ExperimentAnalysis: Object for experiment analysis.
    """

    ray.init(object_store_memory=100000000)
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=10000,
        grace_period=config["grace_period"],
        reduction_factor=2)

    if config["Method"]=='ITES':
        config["alpha"]=0
        result = tune.run(
            partial(fit_BITES, X_train=X_train, Y_train=Y_train, event_train=event_train, treatment_train=treatment_train),
            name=config["Method"]+'_'+config["trial_name"],
            resources_per_trial={"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]},
            config=config,
            num_samples=config["num_samples"],
            scheduler=scheduler,
            checkpoint_at_end=False,
            local_dir=config["result_dir"])

    elif config["Method"]=='BITES':
        result = tune.run(
            partial(fit_BITES, X_train=X_train, Y_train=Y_train, event_train=event_train, treatment_train=treatment_train),
            name=config["Method"]+'_'+config["trial_name"],
            resources_per_trial={"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]},
            config=config,
            num_samples=config["num_samples"],
            scheduler=scheduler,
            checkpoint_at_end=False,
            local_dir=config["result_dir"])


    elif config["Method"]=='DeepSurv':
        if treatment_train is not None:
            config["num_covariates"]=np.c_[treatment_train, X_train].shape[1]
            result = tune.run(
                partial(fit_DeepSurv, X_train=np.c_[treatment_train, X_train], Y_train=Y_train,
                        event_train=event_train),
                name=config["Method"] + '_' + config["trial_name"],
                resources_per_trial={"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]},
                config=config,
                num_samples=config["num_samples"],
                scheduler=scheduler,
                checkpoint_at_end=False,
                local_dir=config["result_dir"])

        else:
            result = tune.run(
                partial(fit_DeepSurv, X_train=X_train, Y_train=Y_train, event_train=event_train),
                name=config["Method"] + '_' + config["trial_name"],
                resources_per_trial={"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]},
                config=config,
                num_samples=config["num_samples"],
                scheduler=scheduler,
                checkpoint_at_end=False,
                local_dir=config["result_dir"])

    elif config["Method"]=='DeepSurvT':
        X_train0, Y_train0, event_train0 = X_train[treatment_train==0], Y_train[treatment_train==0], event_train[treatment_train==0]
        X_train1, Y_train1, event_train1 = X_train[treatment_train==1], Y_train[treatment_train==1],event_train[treatment_train==1]

        result0 = tune.run(
            partial(fit_DeepSurv, X_train=X_train0, Y_train=Y_train0, event_train=event_train0),
            name=config["Method"] + '_T0_' + config["trial_name"],
            resources_per_trial={"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]},
            config=config,
            num_samples=config["num_samples"],
            scheduler=scheduler,
            checkpoint_at_end=False,
            local_dir=config["result_dir"])

        result1 = tune.run(
            partial(fit_DeepSurv, X_train=X_train1, Y_train=Y_train1, event_train=event_train1),
            name=config["Method"] + '_T1_' + config["trial_name"],
            resources_per_trial={"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]},
            config=config,
            num_samples=config["num_samples"],
            scheduler=scheduler,
            checkpoint_at_end=False,
            local_dir=config["result_dir"])

        result=[result0,result1]

    elif config["Method"]=='CFRNet':
        result = tune.run(
            partial(fit_CFRNet, X_train=X_train, Y_train=Y_train, treatment_train=treatment_train),
            name=config["Method"]+'_'+config["trial_name"],
            resources_per_trial={"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]},
            config=config,
            num_samples=config["num_samples"],
            scheduler=scheduler,
            checkpoint_at_end=False,
            local_dir=config["result_dir"])


    else:
        print('Please choose a valid Method!')
        print('bites, DeepSurv, DeepSurvT, CFRNet ')
        return

    return result


def fit_BITES(config, X_train, Y_train, event_train,treatment_train):


    data = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train),
                         torch.Tensor(treatment_train), torch.Tensor(event_train))
    test_abs = int(len(X_train) * (1-config["val_set_fraction"]))
    train_subset, val_subset = random_split(data, [test_abs, len(X_train) - test_abs])

    net = BITES(in_features=X_train.shape[1], num_nodes_shared=config["shared_layer"], num_nodes_indiv=config["individual_layer"], out_features=1,
                         dropout=config["dropout"])

    print(config["individual_layer"])
    device = "cpu"
    pin_memory=False
    if torch.cuda.is_available():
        device = "cuda:0"
        pin_memory = config["pin_memory"]
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,pin_memory=pin_memory,
        num_workers=config["cpus_per_trial"])
    valloader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,pin_memory=pin_memory,
        num_workers=config["cpus_per_trial"])

    best_val_loss = np.Inf
    loss_fkt = BITES_Loss(alpha=config["alpha"], blur=config["blur"])
    loss_fkt_val = BITES_Loss(alpha=0)
    early_stopping_count=0
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        train_loss=0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            x, y, treatment, event = data
            if torch.cuda.is_available():
                x, y, treatment, event = x.cuda(non_blocking=pin_memory), y.cuda(non_blocking=pin_memory), treatment.cuda(non_blocking=pin_memory), event.cuda(non_blocking=pin_memory)
                #x, y, treatment, event = x.to(device), y.to(device), treatment.to(device), event.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = net(x, treatment)
            loss = loss_fkt(y_pred[0], y_pred[1], y,  event, treatment)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss = loss.item()
            epoch_steps += 1


        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                x_val, y_val, treatment_val, event_val = data

                if torch.cuda.is_available():
                    x_val, y_val, treatment_val, event_val = x_val.cuda(non_blocking=pin_memory), y_val.cuda(non_blocking=pin_memory), treatment_val.cuda(non_blocking=pin_memory), event_val.cuda(non_blocking=pin_memory)
                    #x_val, y_val, treatment_val, event_val = x_val.to(device), y_val.to(device), treatment_val.to(device), event_val.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                y_pred_val = net.predict(x_val, treatment_val)

                loss_val = loss_fkt_val(y_pred_val[0], y_pred_val[1], y_val, event_val, treatment_val)
                val_loss += loss_val.cpu().numpy()
                val_steps += 1

        early_stopping_count = early_stopping_count + 1
        if (val_loss / val_steps) < best_val_loss:
            early_stopping_count = 0
            best_val_loss = (val_loss / val_steps)
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(train_loss), val_loss=(val_loss / val_steps))
        if early_stopping_count > config["grace_period"]:
            return
        
    print("Finished Training")
    return


def fit_DeepSurv(config, X_train, Y_train, event_train):

    data = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train), torch.Tensor(event_train))
    test_abs = int(len(X_train) * (1-config["val_set_fraction"]))
    train_subset, val_subset = random_split(data, [test_abs, len(X_train) - test_abs])

    net = DeepSurv(in_features=X_train.shape[1], num_nodes=config["shared_layer"], out_features=1,
                         dropout=config["dropout"])

    device = "cpu"
    pin_memory=False
    if torch.cuda.is_available():
        device = "cuda:0"
        pin_memory = config["pin_memory"]
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,pin_memory=pin_memory,
        num_workers=config["cpus_per_trial"])
    valloader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,pin_memory=pin_memory,
        num_workers=config["cpus_per_trial"])

    best_val_loss = np.Inf
    loss_fkt = DeepSurv_Loss()
    early_stopping_count=0
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        train_loss=0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            x, y, event = data
            if torch.cuda.is_available():
                x, y, event = x.cuda(non_blocking=pin_memory), y.cuda(non_blocking=pin_memory), event.cuda(non_blocking=pin_memory)
                #x, y, treatment, event = x.to(device), y.to(device), treatment.to(device), event.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = net(x)
            loss = loss_fkt(log_h=y_pred, durations=y,  events=event)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss = loss.item()
            epoch_steps += 1


        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                x_val, y_val, event_val = data

                if torch.cuda.is_available():
                    x_val, y_val, event_val = x_val.cuda(non_blocking=pin_memory), y_val.cuda(non_blocking=pin_memory), event_val.cuda(non_blocking=pin_memory)
                    #x_val, y_val, treatment_val, event_val = x_val.to(device), y_val.to(device), treatment_val.to(device), event_val.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                y_pred_val = net.predict(x_val)

                loss_val = loss_fkt(log_h=y_pred_val, durations=y_val,  events=event_val)
                val_loss += loss_val.cpu().numpy()
                val_steps += 1

        early_stopping_count = early_stopping_count + 1
        if (val_loss / val_steps) < best_val_loss:
            early_stopping_count = 0
            best_val_loss = (val_loss / val_steps)
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(train_loss), val_loss=(val_loss / val_steps))
        if early_stopping_count > config["grace_period"]:
            return

    print("Finished Training")
    return


def fit_CFRNet(config, X_train, Y_train, treatment_train):
    data = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train),
                         torch.Tensor(treatment_train))
    test_abs = int(len(X_train) * (1 - config["val_set_fraction"]))
    train_subset, val_subset = random_split(data, [test_abs, len(X_train) - test_abs])

    net = BITES(in_features=X_train.shape[1], num_nodes_shared=config["shared_layer"],
                num_nodes_indiv=config["individual_layer"], out_features=1,
                dropout=config["dropout"])

    print(config["individual_layer"])
    device = "cpu"
    pin_memory = False
    if torch.cuda.is_available():
        device = "cuda:0"
        pin_memory = config["pin_memory"]
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True, pin_memory=pin_memory,
        num_workers=config["cpus_per_trial"])
    valloader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True, pin_memory=pin_memory,
        num_workers=config["cpus_per_trial"])

    best_val_loss = np.Inf
    loss_fkt = CFRNet_Loss(alpha=config["alpha"], blur=config["blur"])
    loss_fkt_val = CFRNet_Loss(alpha=0)
    early_stopping_count = 0
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        train_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            x, y, treatment = data
            if torch.cuda.is_available():
                x, y, treatment = x.cuda(non_blocking=pin_memory), y.cuda(
                    non_blocking=pin_memory), treatment.cuda(non_blocking=pin_memory)
                # x, y, treatment = x.to(device), y.to(device), treatment.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = net(x, treatment)
            loss = loss_fkt(y_pred[0], y_pred[1], y, treatment)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss = loss.item()
            epoch_steps += 1

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                x_val, y_val, treatment_val = data

                if torch.cuda.is_available():
                    x_val, y_val, treatment_val = x_val.cuda(non_blocking=pin_memory), y_val.cuda(
                        non_blocking=pin_memory), treatment_val.cuda(non_blocking=pin_memory)
                    # x_val, y_val, treatment_val = x_val.to(device), y_val.to(device), treatment_val.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                y_pred_val = net.predict(x_val, treatment_val)

                loss_val = loss_fkt_val(y_pred_val[0], y_pred_val[1], y_val, treatment_val)
                val_loss += loss_val.cpu().numpy()
                val_steps += 1

        early_stopping_count = early_stopping_count + 1
        if (val_loss / val_steps) < best_val_loss:
            early_stopping_count = 0
            best_val_loss = (val_loss / val_steps)
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(train_loss), val_loss=(val_loss / val_steps))
        if early_stopping_count > config["grace_period"]:
            return

    print("Finished Training")
    return
