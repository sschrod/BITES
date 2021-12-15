import pickle

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def simulate_data(num_samples=100, shift=0):
    N = 10
    mean = np.zeros(N)
    std_dev = np.ones(N)
    pdf = multivariate_normal(mean, std_dev)

    x1, x2 = np.abs(pdf.rvs(num_samples)),np.abs(pdf.rvs(num_samples))
    X = np.c_[x1, x2]
    gamma1, gamma2 = np.full(N, 0.1), (np.arange(0.01,0.21,0.02)+ 0.005)

    T0 = np.exp((np.dot(x2, gamma1.T) + np.dot(x1, gamma1.T)))
    T1 = np.exp((np.dot(x2, gamma2.T) + np.dot(x1, gamma1.T)))
    T0[T0 > 10] = 10
    T1[T1 > 10] = 10

    pos_treatment_effect = np.sum(T0 - T1 < 0) / num_samples

    best_choice = np.zeros(num_samples)
    best_choice[T0 - T1 < 0] = 1

    """Split into treated and untreated patients"""
    times = np.concatenate((T0[:int(num_samples / 2)], T1[int(num_samples / 2):]))
    times_cf = np.concatenate((T1[:int(num_samples / 2)], T0[int(num_samples / 2):]))
    treatment = np.concatenate((np.zeros(int(num_samples / 2)), np.ones(int(num_samples / 2))))
    df = pd.DataFrame({'times': times, 'times_cf': times_cf, 'treatment': treatment, 'best choice': best_choice})

    """Censor part of the Dataset"""
    percentage_censored = 0.5
    index = np.random.choice(num_samples, int(num_samples * percentage_censored), replace=False)
    times2 = times.copy()
    times2[index] = times[index] * np.random.uniform(size=int(num_samples * percentage_censored))

    censored = np.ones(num_samples)
    censored[index] = 0
    index_10 = np.logical_or(T0 == 10, T1 == 10)
    censored[index_10] = 0
    df['censored'] = censored
    df['times censored'] = times2

    # Add Noise to Covariates N(0,0.1)
    noise_mean = np.zeros(2 * N)
    noise_std_dev = np.ones(2 * N) * 0.1
    noise_pdf = multivariate_normal(noise_mean, noise_std_dev)
    noise = noise_pdf.rvs(num_samples)

    X = X + noise

    Y = df.to_numpy()
    Y = Y.astype('float32')
    X = X.astype('float32')
    return X, Y

def Simulate_data():
    test_data = simulate_data(1000)
    train_data = []

    train_data.append(simulate_data(1000))

    X1, Y1 = simulate_data(1000)
    train_data.append([np.append(train_data[-1][0], X1, axis=0), np.append(train_data[-1][1], Y1, axis=0)])
    X2, Y2 = simulate_data(1000)
    train_data.append([np.append(train_data[-1][0], X2, axis=0), np.append(train_data[-1][1], Y2, axis=0)])
    X3, Y3 = simulate_data(1000)
    train_data.append([np.append(train_data[-1][0], X3, axis=0), np.append(train_data[-1][1], Y3, axis=0)])


    pickle.dump(test_data, open('test_data.Sim1', 'wb'))
    pickle.dump(train_data, open('train_data.Sim1', 'wb'))

    return

if __name__ == '__main__':
    Simulate_data()