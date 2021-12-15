import numpy as np

from bites.analyse.analyse_utils import *
from bites.data.RGBSG.RGBSG_utilis import load_RGBSG, load_RGBSG_no_onehot



if __name__=='__main__':

    """Choose Method to analyse!"""
    method='BITES'
    compare_against_ATE=True



    X_train, Y_train, event_train, treatment_train, _, _ = load_RGBSG(partition='train',
                                                                      filename_="./data/RGBSG/rgbsg.h5")
    X_test, Y_test, event_test, treatment_test, _, _ = load_RGBSG(partition='test',
                                                                  filename_="./data/RGBSG/rgbsg.h5")

    if method=='BITES' or method=='ITES':
        model,config=get_best_model("ray_results/"+method+"_RGBSG")
        model.compute_baseline_hazards(X_train,[Y_train, event_train, treatment_train])

        C_index,_,_=get_C_Index_BITES(model, X_test, Y_test, event_test, treatment_test)
        pred_ite,_=get_ITE_BITES(model,X_test,treatment_test)

        if compare_against_ATE:
            sns.set(font_scale=1)
            analyse_randomized_test_set(np.ones_like(pred_ite), Y_test, event_test, treatment_test, C_index=C_index, method_name=None,save_path=None,annotate=False)
            analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test,C_index=C_index,method_name=method, save_path='../../../Dokumente/Paper/Plots/RGBSG_'+method+'_baseline.pdf',new_figure=False,annotate=True)
        else:
            analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=C_index,
                                        method_name=method,
                                        save_path='../../../Dokumente/Paper/Plots/RGBSG_' + method + '.pdf')
    elif method=='DeepSurvT':
        model0, config0 = get_best_model("ray_results/" + method + "_T0_RGBSG",assign_treatment=0)
        model0.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])

        model1, config1 = get_best_model("ray_results/" + method + "_T1_RGBSG",assign_treatment=1)
        model1.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])

        C_index, _, _ =get_C_Index_DeepSurvT(model0,model1,X_test,Y_test,event_test,treatment_test)

        pred_ite,_=get_ITE_DeepSurvT(model0, model1, X_test, treatment_test, best_treatment=None, death_probability=0.5)

        sns.set(font_scale=1)
        analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test, C_index=C_index, method_name=method,
                                    save_path='../../../Dokumente/Paper/Plots/RGBSG_' + method + '.pdf')




    do_SHAP=False
    if do_SHAP and method=='BITES':
        import shap
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        model.eval()


        def net_treatment0(input):
            ohc = OneHotEncoder(sparse=False)
            X_ohc = ohc.fit_transform(input[:, -2:])
            tmp=np.c_[input[:,:-2],X_ohc].astype('float32')
            return model.risk_nets[0](model.shared_net(torch.tensor(tmp))).detach().numpy()


        def net_treatment1(input):
            ohc = OneHotEncoder(sparse=False)
            X_ohc = ohc.fit_transform(input[:, -2:])
            tmp=np.c_[input[:,:-2],X_ohc].astype('float32')
            return model.risk_nets[1](model.shared_net(torch.tensor(tmp))).detach().numpy()

        """Load data without one_hot encoding"""

        X_train, Y_train, event_train, treatment_train, _ = load_RGBSG_no_onehot(partition='train',
                                                                          filename_="./BITES/data/RGBSG/rgbsg.h5")

        X_test, Y_test, event_test, treatment_test, _ = load_RGBSG_no_onehot(partition='test',
                                                                      filename_="./BITES/data/RGBSG/rgbsg.h5")



        X_train0 = X_train[treatment_train == 0]
        X_train1 = X_train[treatment_train == 1]
        names = ['N pos nodes', 'Age', 'Progesterone', 'Estrogene','Menopause', 'Grade']
        X_test0 = pd.DataFrame(X_test[treatment_test == 0], columns=names)
        X_test1 = pd.DataFrame(X_test[treatment_test == 1], columns=names)

        explainer_treatment0 = shap.Explainer(net_treatment0, X_train0)
        explainer_treatment1 = shap.Explainer(net_treatment1, X_train1)

        shap_values0_temp = explainer_treatment0(X_test0.astype('float32'))
        shap_values1_temp = explainer_treatment1(X_test1.astype('float32'))

        #temp = open("order.plk", "rb")
        #order = pickle.load(temp)

        plt.style.use('default')
        fig, ax = plt.subplots(1, 2)
        plt.axes(ax[0])
        #shap.plots.beeswarm(shap_values0_temp, color_bar_label=None, color_bar=None, order=order.abs.mean(0))
        shap.plots.beeswarm(shap_values0_temp, color_bar_label=None, color_bar=None)
        ax[0].set_xlabel('SHAP value')
        ax[0].set_xlim([-0.5, 1])
        ax[0].annotate('a', xy=(0.02, 0.92), xycoords='axes fraction', fontsize='x-large')
        plt.title('No Hormone Treatment')

        plt.axes(ax[1])
        #shap.plots.beeswarm(shap_values1_temp, order=order.abs.mean(0))
        shap.plots.beeswarm(shap_values1_temp,order=shap_values0_temp.abs.mean(0),color_bar_label=None, color_bar=None)
        plt.title('Hormone Treatment')
        ax[1].set_xlabel('SHAP value')
        ax[1].set_yticks([])
        ax[1].set_xlim([-0.5, 1])
        ax[1].annotate('b', xy=(0.02, 0.92), xycoords='axes fraction', fontsize='x-large')
        fig.tight_layout()
        plt.savefig('../../../Dokumente/Paper/Plots/SHAP_RGBSG_BITES.pdf', format='pdf')