import numpy as np


def theta_transform(data):
    """
    Input
    -----
    I_train : np.ndarray
        Clean intensity data for training set (left unchanged).
    theta_train : np.ndarray
        Raw theta for training set, shape [N_train, D].

    Output
    ------
    I_train : np.ndarray
        Same object/values as input (unchanged).
    theta_tilde : np.ndarray
        Standardized theta: (theta - mu) / s, shape [N_train, D].
    mu : np.ndarray
        Mean of theta_train over training set, shape [D,].
    s : np.ndarray
        Std of theta_train over training set (ddof=0), shape [D,].
    """
    
    # if len(data) == 2:
    I_train, c_max_train = data[0], data[1]

    mu = c_max_train.mean(axis=0)
    s = c_max_train.std(axis=0, ddof=0)

    c_max_train_tilde = (c_max_train - mu) / s

    return I_train, c_max_train_tilde, mu, s
    
    # elif len(data) == 3:
    #     I_train, theta_condition, theta_inference = data[0], data[1], data[2]

    #     mu_condition = theta_condition.mean(axis=0)
    #     s_condition = theta_condition.std(axis=0, ddof=0)

    #     mu_inference = theta_inference.mean(axis=0)
    #     s_inference = theta_inference.std(axis=0, ddof=0)

    #     theta_condition_tilde = (theta_condition - mu_condition) / s_condition
    #     theta_inference_tilde = (theta_inference - mu_inference) / s_inference

    #     return I_train, theta_condition_tilde, mu_condition, s_condition, theta_inference_tilde, mu_inference, s_inference