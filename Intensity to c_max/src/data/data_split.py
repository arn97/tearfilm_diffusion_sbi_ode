import numpy as np


def split_dataset(
    data,
    train_frac=0.7,
    test_frac=0.2,
    val_frac=0.1,
    shuffle=True,
    seed=42,
):
    # if len(data) == 2:
    I, c_max = data
    # if len(data) == 3:
    #     I, theta_condition, theta_inference = data
        
    if not np.isclose(train_frac + test_frac + val_frac, 1.0):
        raise ValueError("Fractions must sum to 1.")

    N = I.shape[0]

    indices = np.arange(N)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(train_frac * N)
    n_test = int(test_frac * N)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]
    val_idx = indices[n_train + n_test:]

    # if len(data) == 2:
    return (
        (I[train_idx], c_max[train_idx]),
        (I[test_idx], c_max[test_idx]),
        (I[val_idx], c_max[val_idx]),
    )
    # if len(data) == 3:
    #     return (
    #         (I[train_idx], theta_condition[train_idx], theta_inference[train_idx]),
    #         (I[test_idx], theta_condition[test_idx], theta_inference[test_idx]),
    #         (I[val_idx], theta_condition[val_idx], theta_inference[val_idx]),
    #     )