import numpy as np
from bites.analyse.analyse_utils import *
import pickle

if __name__ == '__main__':

    """Choose Method to analyse!"""
    ###############################
    method = 'BITES'                    # Set Name: BITES, ITES, DeepSurv, DeepSurvT, CFRNet
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
    if method == 'BITES' or method == 'ITES':
        model, config = get_best_model(result_path)
        model.compute_baseline_hazards(X_train, [y_train[:, 5], y_train[:, 4], y_train[:, 2]])

        C_index, C_index_T0, C_index_T1 = get_C_Index_BITES(model, X_test, Y_test, event_test, treatment_test)
        pred_ite, correct_predicted_probability = get_ITE_BITES(model, X_test, treatment_test, best_treatment = y_test[:,3])
