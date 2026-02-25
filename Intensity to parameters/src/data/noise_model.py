import numpy as np


def add_multiplicative_noise(data, noise_std=0.01, seed=None):
    """
    Inject multiplicative Gaussian noise into the intensity component of a dataset.

    Noise model:
        x = I * (1 + noise_std * z),  z ~ N(0,1)

    Accepted inputs
    ---------------
    Training dataset tuple:
        (I_train, theta_tilde, mu, s)
        -> returns (x_train, theta_tilde, mu, s)

    Test/Val dataset tuple:
        (I_split, theta_split)
        -> returns (x_split, theta_split)

    Parameters
    ----------
    data : tuple
        Either (I, theta, mu, s) or (I, theta)
    noise_std : float
        Standard deviation for multiplicative noise (default 0.01)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    tuple
        Same structure as input, but with I replaced by x.
    """
    if not isinstance(data, tuple):
        raise TypeError("data must be a tuple.")

    if len(data) == 4:
        I, theta, mu, s = data
    elif len(data) == 2:
        I, theta = data
    if len(data) == 7:
        I, theta_condition, mu_condition, s_condition, theta_inference, mu_inference, s_inference = data
    elif len(data) == 3:
        I, theta_condition, theta_inference = data
    else:
        raise ValueError("data must be a tuple of length 4 (train) or 2 (test/val) or 7 (train with both condition and inference) or 3 (test/val with both condition and inference).")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=I.shape)
    x = I * (1.0 + noise_std * z)

    x = x.astype(np.float32, copy=False)

    if len(data) == 4:
        return (x, theta, mu, s)
    elif len(data) == 7:
        return (x, theta_condition, mu_condition, s_condition, theta_inference, mu_inference, s_inference)
    elif len(data) == 3:
        return (x, theta_condition, theta_inference)
    else:
        return (x, theta)