import numpy as np
from bites.analyse.analyse_utils import *
from data.RGBSG.RGBSG_utilis import load_RGBSG

if __name__ == '__main__':

    """Choose Method to analyse!"""
    method = 'BITES'
    compare_against_ATE = False

    X_train, Y_train, event_train, treatment_train, _, _ = load_RGBSG(partition='train',
                                                                      filename_="./data/RGBSG/rgbsg.h5")
    X_test, Y_test, event_test, treatment_test, _, _ = load_RGBSG(partition='test',
                                                                  filename_="./data/RGBSG/rgbsg.h5")

    if method == 'BITES' or method == 'ITES':
        model, config = get_best_model("ray_results/" + method + "_RGBSG")
        model.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])

        C_index, _, _ = get_C_Index_BITES(model, X_test, Y_test, event_test, treatment_test)
        pred_ite, _ = get_ITE_BITES(model, X_test, treatment_test)

        if compare_against_ATE:
            sns.set(font_scale=1)
            analyse_randomized_test_set(np.ones_like(pred_ite), Y_test, event_test, treatment_test, C_index=C_index,
                                        method_name=None, save_path=None, annotate=False)
            analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=C_index,
                                        method_name=method, save_path='RGBSG_' + method + '_baseline.pdf',
                                        new_figure=False, annotate=True)
        else:
            analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=C_index,
                                        method_name=method,
                                        save_path='RGBSG_' + method + '.pdf')
    elif method == 'DeepSurvT':
        model0, config0 = get_best_model("ray_results/" + method + "_T0_RGBSG", assign_treatment=0)
        model0.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])

        model1, config1 = get_best_model("ray_results/" + method + "_T1_RGBSG", assign_treatment=1)
        model1.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])

        C_index, _, _ = get_C_Index_DeepSurvT(model0, model1, X_test, Y_test, event_test, treatment_test)

        pred_ite, _ = get_ITE_DeepSurvT(model0, model1, X_test, treatment_test, best_treatment=None,
                                        death_probability=0.5)

        sns.set(font_scale=1)
        analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=C_index, method_name=method,
                                    save_path='RGBSG_' + method + '.pdf')

    elif method == 'DeepSurv':
        model0, config0 = get_best_model("ray_results/" + method + "_RGBSG")
        model0.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])

        C_index, _, _ = get_C_Index_DeepSurv(model0, X_test, Y_test, event_test, treatment_test)
