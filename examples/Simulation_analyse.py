import numpy as np
from bites.analyse.analyse_utils import *
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':

    """Choose Method to analyse!"""
    ###############################
    method = 'DeepSurv'                    # Set Name: BITES, ITES, DeepSurv, DeepSurvT, CFRNet
    results_dir='ray_results/'          # Set result_dir
    num_training_samples = 0            # int 0 to 4 for 1000 to 4000 training samples
    trial_name = 'Simulation3'          # Name used in the config
    ################################

    """Load triaining (for baseline hazards) and test data."""
    X_test, y_test = pickle.load(open('data/Simulation_Treatment_Bias/test_data.Sim3', 'rb'))
    X_train, y_train = pickle.load(open('data/Simulation_Treatment_Bias/train_data.Sim3', 'rb'))[0]
    Y_test, event_test, treatment_test = y_test[:, 5], y_test[:, 4], y_test[:, 2]

    # Analysis of the different Methods
    result_path = results_dir + method + "_" + trial_name
    pred_ite=None
    if method == 'BITES' or method == 'ITES':
        model, config = get_best_model(result_path)
        model.compute_baseline_hazards(X_train, [y_train[:, 5], y_train[:, 4], y_train[:, 2]])

        C_index, C_index_T0, C_index_T1 = get_C_Index_BITES(model, X_test, Y_test, event_test, treatment_test)
        pred_ite, correct_predicted_probability = get_ITE_BITES(model, X_test, treatment_test, best_treatment = y_test[:,3])

    elif method == 'DeepSurvT':
        model0, config0 = get_best_model(results_dir + method + "_T0_" + trial_name, assign_treatment=0)
        model0.compute_baseline_hazards(X_train, [y_train[:, 5], y_train[:, 4], y_train[:, 2]])

        model1, config1 = get_best_model(results_dir + method + "_T1_" + trial_name, assign_treatment=1)
        model1.compute_baseline_hazards(X_train, [y_train[:, 5], y_train[:, 4], y_train[:, 2]])

        C_index, C_index_T0, C_index_T1 = get_C_Index_DeepSurvT(model0, model1, X_test, Y_test, event_test, treatment_test)

        pred_ite, correct_predicted_probability = get_ITE_DeepSurvT(model0, model1, X_test, treatment_test, best_treatment=y_test[:,3],
                                        death_probability=0.5)

    elif method == 'DeepSurv':
        treatment_train = y_train[:, 2]
        if treatment_train is not None:
            X_train=np.c_[X_train, treatment_train]
            X_test=np.c_[X_test, treatment_test]
            model, config = get_best_model(result_path)
            model.compute_baseline_hazards(X_train, [y_train[:, 5], y_train[:, 4], y_train[:, 2]])
            C_index, C_index_T0, C_index_T1 = get_C_Index_DeepSurv(model, X_test, Y_test, event_test)
            pred_ite, correct_predicted_probability = get_ITE_DeepSurv(model, X_test, treatment_test, best_treatment=y_test[:,3], death_probability=0.5)

        else:
            print("No treatment set, return best possible model")
            model, config = get_best_model(result_path)
            model.compute_baseline_hazards(X_train, [y_train[:, 5], y_train[:, 4], None])

    elif method == 'CFRNet':
        model, config = get_best_model(result_path)
        pred_ite = get_ITE_CFRNet(model, X_test, treatment_test, best_treatment=None)


    else:
        print(method+' Not defined!')

    if pred_ite is not None:
        plot_ITE_correlation(pred_ite, y_true=y_test[:,0],y_cf=y_test[:,1],treatment=treatment_test)