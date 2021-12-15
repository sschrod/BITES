import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_RGBSG(partition='complete',filename_ = r"""rgbsg.h5"""):
    """References
    ----------
    .. [#katzman0] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    """
    # Read training data.
    with h5py.File(filename_, 'r') as f:
        X_train = f['train']['x'][()]
        E_train = f['train']['e'][()]
        Y_train = f['train']['t'][()].reshape(-1, 1)

    # Read testing data.
    with h5py.File(filename_, 'r') as f:
        X_test = f['test']['x'][()]
        E_test = f['test']['e'][()]
        Y_test = f['test']['t'][()].reshape(-1, 1)

    # Define data partitions.
    if partition == 'training' or partition == 'train':
        X = X_train
        Y = Y_train
        event = E_train
    elif partition == 'testing' or partition == 'test':
        X = X_test
        Y = Y_test
        event = E_test
    elif partition == 'complete':
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        event = np.concatenate((E_train, E_test), axis=0)
    else:
        raise ValueError('Invalid partition.')

    column_names = ['horm_treatment', 'grade', 'menopause', 'age', 'n_positive_nodes', 'progesterone', 'estrogen']
    df_X = pd.DataFrame(data=X, columns=column_names)
    df_X=df_X[['horm_treatment', 'menopause' , 'grade', 'n_positive_nodes', 'age', 'progesterone', 'estrogen']]

    scaler=StandardScaler()
    X_std=scaler.fit_transform(df_X.values[:,3:])

    ohc = OneHotEncoder(sparse=False)
    X_ohc=ohc.fit_transform(df_X.values[:,1:3])

    X=np.c_[X_std,X_ohc]
    treatment=df_X['horm_treatment'].to_numpy().flatten()

    return X, Y.flatten(), event.flatten(),treatment, scaler, ohc


def load_RGBSG_no_onehot(partition='complete',filename_ = r"""rgbsg.h5"""):
    """References
    ----------
    .. [#katzman0] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    """
    # Read training data.
    with h5py.File(filename_, 'r') as f:
        X_train = f['train']['x'][()]
        E_train = f['train']['e'][()]
        Y_train = f['train']['t'][()].reshape(-1, 1)

    # Read testing data.
    with h5py.File(filename_, 'r') as f:
        X_test = f['test']['x'][()]
        E_test = f['test']['e'][()]
        Y_test = f['test']['t'][()].reshape(-1, 1)

    # Define data partitions.
    if partition == 'training' or partition == 'train':
        X = X_train
        Y = Y_train
        event = E_train
    elif partition == 'testing' or partition == 'test':
        X = X_test
        Y = Y_test
        event = E_test
    elif partition == 'complete':
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        event = np.concatenate((E_train, E_test), axis=0)
    else:
        raise ValueError('Invalid partition.')

    column_names = ['horm_treatment', 'grade', 'menopause', 'age', 'n_positive_nodes', 'progesterone', 'estrogen']
    df_X = pd.DataFrame(data=X, columns=column_names)
    df_X=df_X[['horm_treatment', 'menopause' , 'grade', 'n_positive_nodes', 'age', 'progesterone', 'estrogen']]

    scaler=StandardScaler()
    X_std=scaler.fit_transform(df_X.values[:,3:])

    X=np.c_[X_std,df_X.values[:,1:3]]
    treatment=df_X['horm_treatment'].to_numpy().flatten()

    return X, Y.flatten(), event.flatten(),treatment, scaler

