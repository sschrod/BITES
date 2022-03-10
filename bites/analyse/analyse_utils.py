import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bites.model.BITES_base import BITES
from bites.model.CFRNet_base import CFRNet
from bites.model.DeepSurv_base import DeepSurv
from bites.utils.eval_surv import EvalSurv
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from ray.tune import Analysis


def get_best_model(path_to_experiment="./ray_results/test_hydra", assign_treatment=None):
    analysis = Analysis(path_to_experiment, default_metric="val_loss", default_mode="min")
    best_config = analysis.get_best_config()
    best_checkpoint_dir = analysis.get_best_checkpoint(analysis.get_best_logdir())

    if best_config["Method"] == 'BITES' or best_config["Method"] == 'ITES':
        best_net = BITES(best_config["num_covariates"], best_config["shared_layer"], best_config["individual_layer"],
                         out_features=1,
                         dropout=best_config["dropout"])

    elif best_config["Method"] == 'DeepSurv' or best_config["Method"] == 'DeepSurvT':
        best_net = DeepSurv(best_config["num_covariates"], best_config["shared_layer"], out_features=1,
                            dropout=best_config["dropout"])
        best_net.treatment = assign_treatment

    elif best_config["Method"] == 'CFRNet':
        best_net = CFRNet(best_config["num_covariates"], best_config["shared_layer"], best_config["individual_layer"],
                         out_features=1,
                         dropout=best_config["dropout"])

    else:
        print('Method not implemented yet!')
        return

    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"), map_location=torch.device('cpu'))

    best_net.load_state_dict(model_state)

    return best_net, best_config


def get_C_Index_BITES(model, X, time, event, treatment):
    if not model.baseline_hazards_:
        print('Compute Baseline Hazards before running get_C_index')
        return

    surv0, surv1 = model.predict_surv_df(X, treatment)
    surv = pd.concat([surv0, surv1], axis=1)
    surv = surv.interpolate('index')
    C_index0 = EvalSurv(surv0, time[treatment == 0], event[treatment == 0], censor_surv='km').concordance_td()
    C_index1 = EvalSurv(surv1, time[treatment == 1], event[treatment == 1], censor_surv='km').concordance_td()
    C_index = EvalSurv(surv, np.append(time[treatment == 0], time[treatment == 1]),
                       np.append(event[treatment == 0], event[treatment == 1]),
                       censor_surv='km').concordance_td()

    print('Time dependent C-Index: ' + str(C_index)[:5])
    print('Treatment 0 C-Index: ' + str(C_index0)[:5])
    print('Treatment 1 C-Index: ' + str(C_index1)[:5])

    return C_index, C_index0, C_index1


def get_C_Index_DeepSurvT(model0, model1, X, time, event, treatment):

    mask0 = treatment == 0
    mask1 = treatment == 1

    X0, time0, event0 = X[mask0], time[mask0], event[mask0]
    X1, time1, event1 = X[mask1], time[mask1], event[mask1]
    surv0 = model0.predict_surv_df(X0)
    surv1 = model1.predict_surv_df(X1)

    surv = pd.concat([surv0, surv1], axis=1)
    surv = surv.interpolate('index')
    C_index = EvalSurv(surv, np.append(time0, time1),
                       np.append(event0, event1), censor_surv='km').concordance_td()
    C_index0 = EvalSurv(surv0, time0, event0, censor_surv='km').concordance_td()
    C_index1 = EvalSurv(surv1, time1, event1, censor_surv='km').concordance_td()

    print('Time dependent C-Index: ' + str(C_index)[:5])
    print('Treatment 0 C-Index: ' + str(C_index0)[:5])
    print('Treatment 1 C-Index: ' + str(C_index1)[:5])

    return C_index, C_index0, C_index1

def get_C_Index_DeepSurv(model, X, time, event, treatment=None):
    if treatment is not None:
        surv = model.predict_surv_df(np.c_[treatment,X])
        C_index = EvalSurv(surv, time, event, censor_surv='km').concordance_td()
        print('Time dependent C-Index: ' + str(C_index)[:5])
    else:
        surv = model.predict_surv_df(X)
        C_index = EvalSurv(surv, time, event, censor_surv='km').concordance_td()
        print('Time dependent C-Index: ' + str(C_index)[:5])
    return C_index, None, None


def get_ITE_BITES(model, X, treatment, best_treatment=None, death_probability=0.5):
    if not model.baseline_hazards_:
        print('Compute Baseline Hazards before running get_ITE()')
        return

    def find_nearest_index(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    surv0, surv1 = model.predict_surv_df(X, treatment)
    surv0_cf, surv1_cf = model.predict_surv_counterfactual_df(X, treatment)

    """Find factual and counterfactual prediction: Value at 50% survival probability"""
    pred0 = np.zeros(surv0.shape[1])
    pred0_cf = np.zeros(surv0.shape[1])
    for i in range(surv0.shape[1]):
        pred0[i] = surv0.axes[0][find_nearest_index(surv0.iloc[:, i].values, death_probability)]
        pred0_cf[i] = surv0_cf.axes[0][find_nearest_index(surv0_cf.iloc[:, i].values, death_probability)]
    ITE0 = pred0_cf - pred0

    pred1 = np.zeros(surv1.shape[1])
    pred1_cf = np.zeros(surv1.shape[1])
    for i in range(surv1.shape[1]):
        pred1[i] = surv1.axes[0][find_nearest_index(surv1.iloc[:, i].values, death_probability)]
        pred1_cf[i] = surv1_cf.axes[0][find_nearest_index(surv1_cf.iloc[:, i].values, death_probability)]
    ITE1 = pred1 - pred1_cf

    ITE = np.zeros(X.shape[0])
    k, j = 0, 0
    for i in range(X.shape[0]):
        if treatment[i] == 0:
            ITE[i] = ITE0[k]
            k = k + 1
        else:
            ITE[i] = ITE1[j]
            j = j + 1

    correct_predicted_probability=None
    if best_treatment is not None:
        correct_predicted_probability=np.sum(best_treatment==(ITE>0)*1)/best_treatment.shape[0]
        print('Fraction best choice: ' + str(correct_predicted_probability))

    return ITE, correct_predicted_probability

def get_ITE_CFRNet(model, X, treatment, best_treatment=None):

    pred,_ = model.predict_numpy(X, treatment)
    pred_cf,_ = model.predict_numpy(X, 1-treatment)

    ITE = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if treatment[i] == 0:
            ITE[i] = pred_cf[i]-pred[i]
        else:
            ITE[i] = pred[i]-pred_cf[i]

    correct_predicted_probability=None
    if best_treatment is not None:
        correct_predicted_probability=np.sum(best_treatment==(ITE>0)*1)/best_treatment.shape[0]
        print('Fraction best choice: ' + str(correct_predicted_probability))

    return ITE, correct_predicted_probability




def get_ITE_DeepSurvT(model0, model1, X, treatment, best_treatment=None, death_probability=0.5):
    def find_nearest_index(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    mask0 = treatment == 0
    mask1 = treatment == 1

    X0 = X[mask0]
    X1 = X[mask1]
    surv0 = model0.predict_surv_df(X0)
    surv0_cf = model1.predict_surv_df(X0)
    surv1 = model1.predict_surv_df(X1)
    surv1_cf = model0.predict_surv_df(X1)

    """Find factual and counterfactual prediction: Value at 50% survival probability"""
    pred0 = np.zeros(surv0.shape[1])
    pred0_cf = np.zeros(surv0.shape[1])
    for i in range(surv0.shape[1]):
        pred0[i] = surv0.axes[0][find_nearest_index(surv0.iloc[:, i].values, death_probability)]
        pred0_cf[i] = surv0_cf.axes[0][find_nearest_index(surv0_cf.iloc[:, i].values, death_probability)]
    ITE0 = pred0_cf - pred0

    pred1 = np.zeros(surv1.shape[1])
    pred1_cf = np.zeros(surv1.shape[1])
    for i in range(surv1.shape[1]):
        pred1[i] = surv1.axes[0][find_nearest_index(surv1.iloc[:, i].values, death_probability)]
        pred1_cf[i] = surv1_cf.axes[0][find_nearest_index(surv1_cf.iloc[:, i].values, death_probability)]
    ITE1 = pred1 - pred1_cf

    ITE = np.zeros(X.shape[0])
    k, j = 0, 0
    for i in range(X.shape[0]):
        if treatment[i] == 0:
            ITE[i] = ITE0[k]
            k = k + 1
        else:
            ITE[i] = ITE1[j]
            j = j + 1

    correct_predicted_probability=None
    if best_treatment is not None:
        correct_predicted_probability=np.sum(best_treatment==(ITE>0)*1)/best_treatment.shape[0]
        print('Fraction best choice: ' + str(correct_predicted_probability))

    return ITE, correct_predicted_probability

def get_ITE_DeepSurv(model, X, treatment, best_treatment=None, death_probability=0.5):
    def find_nearest_index(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    mask0 = treatment == 0
    mask1 = treatment == 1

    X0 = X[mask0]
    X1 = X[mask1]
    surv0 = model.predict_surv_df(X0)
    surv0_cf = model.predict_surv_df(np.c_[1-X0[:,0],X0[:,1:]])
    surv1 = model.predict_surv_df(X1)
    surv1_cf = model.predict_surv_df(np.c_[1-X1[:,0],X1[:,1:]])

    """Find factual and counterfactual prediction: Value at 50% survival probability"""
    pred0 = np.zeros(surv0.shape[1])
    pred0_cf = np.zeros(surv0.shape[1])
    for i in range(surv0.shape[1]):
        pred0[i] = surv0.axes[0][find_nearest_index(surv0.iloc[:, i].values, death_probability)]
        pred0_cf[i] = surv0_cf.axes[0][find_nearest_index(surv0_cf.iloc[:, i].values, death_probability)]
    ITE0 = pred0_cf - pred0

    pred1 = np.zeros(surv1.shape[1])
    pred1_cf = np.zeros(surv1.shape[1])
    for i in range(surv1.shape[1]):
        pred1[i] = surv1.axes[0][find_nearest_index(surv1.iloc[:, i].values, death_probability)]
        pred1_cf[i] = surv1_cf.axes[0][find_nearest_index(surv1_cf.iloc[:, i].values, death_probability)]
    ITE1 = pred1 - pred1_cf

    ITE = np.zeros(X.shape[0])
    k, j = 0, 0
    for i in range(X.shape[0]):
        if treatment[i] == 0:
            ITE[i] = ITE0[k]
            k = k + 1
        else:
            ITE[i] = ITE1[j]
            j = j + 1

    correct_predicted_probability=None
    if best_treatment is not None:
        correct_predicted_probability=np.sum(best_treatment==(ITE>0)*1)/best_treatment.shape[0]
        print('Fraction best choice: ' + str(correct_predicted_probability))

    return ITE, correct_predicted_probability


def analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=None, method_name='set_name', save_path=None,new_figure=True,annotate=True):
    mask_recommended = (pred_ite > 0) == treatment_test
    mask_antirecommended = (pred_ite < 0) == treatment_test

    recommended_times = Y_test[mask_recommended]
    recommended_event = event_test[mask_recommended]
    antirecommended_times = Y_test[mask_antirecommended]
    antirecommended_event = event_test[mask_antirecommended]

    logrank_result = logrank_test(recommended_times, antirecommended_times, recommended_event, antirecommended_event, alpha=0.95)

    colors = sns.color_palette()
    kmf = KaplanMeierFitter()
    kmf_cf = KaplanMeierFitter()
    if method_name==None:
        kmf.fit(recommended_times, recommended_event, label='Treated')
        kmf_cf.fit(antirecommended_times, antirecommended_event, label='Control')
    else:
        kmf.fit(recommended_times, recommended_event, label=method_name + ' Recommendation')
        kmf_cf.fit(antirecommended_times, antirecommended_event, label=method_name + ' Anti-Recommendation')


    if new_figure:
        #plt.figure(figsize=(8, 2.7))
        #kmf.plot(c=colors[0])
        #kmf_cf.plot(c=colors[1])
        if method_name==None:
            kmf.plot(c=colors[0],ci_show=False)
            kmf_cf.plot(c=colors[1],ci_show=False)
        else:
            kmf.plot(c=colors[0])
            kmf_cf.plot(c=colors[1])
    else:
        kmf.plot(c=colors[2])
        kmf_cf.plot(c=colors[3])


    if annotate:
        # Calculate p-value text position and display.
        y_pos = 0.4
        plt.text(1 * 3, y_pos, f"$p$ = {logrank_result.p_value:.6f}", fontsize='small')
        fraction2 = np.sum((pred_ite > 0)) / pred_ite.shape[0]
        plt.text(1 * 3, 0.3, 'C-Index=' + str(C_index)[:5], fontsize='small')
        plt.text(1 * 3, 0.2, f"{fraction2 * 100:.1f}% recommended for T=1", fontsize='small')

    plt.xlabel('Survival Time [month]')
    plt.ylabel('Survival Probability')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf')


def plot_ITE_correlation(pred_ITE, y_true,y_cf,treatment):
    ITE = np.zeros(pred_ITE.shape[0])
    true_ITE0 = -(y_true[treatment == 0] - y_cf[treatment == 0])
    true_ITE1 = y_true[treatment == 1] - y_cf[treatment == 1]
    k, j = 0, 0
    for i in range(pred_ITE.shape[0]):
        if treatment[i] == 0:
            ITE[i] = true_ITE0[k]
            k = k + 1
        else:
            ITE[i] = true_ITE1[j]
            j = j + 1

    ax=sns.scatterplot(x=ITE,y=pred_ITE)
    ax.set(xlabel='ITE', ylabel='pred_ITE')