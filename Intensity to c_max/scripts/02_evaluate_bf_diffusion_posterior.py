# 02_evaluate_bf_diffusion_posterior.py

import sys
import os
import keras
import numpy as np
import bayesflow as bf
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "keras"

# ------------------------------------------------------------------
# Add project paths
# ------------------------------------------------------------------
DATA_PATH = r"C:\Users\arnab\OneDrive\Desktop\Study material\Research\tearfilm_diffusion_sbi\src\data"
BF_PATH   = r"C:\Users\arnab\OneDrive\Desktop\Study material\Research\tearfilm_diffusion_sbi\src\bf"

sys.path.append(DATA_PATH)
sys.path.append(BF_PATH)

# ------------------------------------------------------------------
# Import your existing scripts
# ------------------------------------------------------------------
from h5_dataset import load_dataset
from data_split import split_dataset
from theta_transform import theta_transform
from noise_model import add_multiplicative_noise

np.random.seed(0)

from make_workflow import make_workflow


# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
I_clean, theta = load_dataset(downsample_to=101)

splits = split_dataset(I_clean, theta, train_frac=0.7, test_frac=0.2, val_frac=0.1, shuffle=True, seed=42)

I_train, theta_train = splits[0]
I_test, theta_test = splits[1]
I_val, theta_val = splits[2]

I_train_clean, theta_tilde, mu, s = theta_transform(splits[0])

x_train, theta_tilde_out, mu_out, s_out = add_multiplicative_noise(
    (I_train_clean, theta_tilde, mu, s),
    noise_std=0.01,
    seed=123
)

x_test, theta_test_out = add_multiplicative_noise(
    (I_test, theta_test),
    noise_std=0.01,
    seed=456
)

theta_tilde_test_out = (theta_test_out - mu_out) / s_out  # standardize test theta using train mu and s

x_val, theta_val_out = add_multiplicative_noise(
    (I_val, theta_val),
    noise_std=0.01,
    seed=789
)

theta_tilde_val_out = (theta_val_out - mu_out) / s_out  # standardize val theta using train mu and s

train_dict = {"x": x_train, "theta": theta_tilde_out}
test_dict = {"x": x_test, "theta": theta_tilde_test_out}
val_dict = {"x": x_val, "theta": theta_tilde_val_out}

# ------------------------------------------------------------------
# 2. Create diffusion model + workflow
# ------------------------------------------------------------------
diffnet, workflow = make_workflow(
    widths=(512, 1024, 2048, 2048, 1024, 512, 256),
    time_embedding_dim=128,
    diffusion_steps=1000,
)


# ------------------------------------------------------------------
# 3. Override learning rate (simple control)
# ------------------------------------------------------------------
# workflow.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# ------------------------------------------------------------------
# 3. Load trained model weights
# ------------------------------------------------------------------
model_path = r"C:\Users\arnab\OneDrive\Desktop\Study material\Research\tearfilm_diffusion_sbi\checkpoints\tearfilm_sbi_diffusion.keras\Tear_film_SBI_model_1.keras"
train_after_load = False  # Set to True if you want to continue training after loading the model

if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}. Training a new model...")
    _ = workflow.fit_offline(
        train_dict,
        validation_data=val_dict,
        epochs=20,
        batch_size=256,
        verbose=2,
    )
    workflow.approximator.save(model_path)
elif train_after_load:
    print(f"Model file found at {model_path}. Loading model and continuing training...")
    workflow.approximator = keras.models.load_model(model_path)
    _ = workflow.fit_offline(
        train_dict,
        validation_data=val_dict,
        epochs=2,
        batch_size=256,
        verbose=2,
    )
    workflow.approximator.save(model_path)
else:
    print(f"Model file found at {model_path}. Loading model without further training...")
    workflow.approximator = keras.models.load_model(model_path)

diagnostics = workflow.plot_default_diagnostics(test_data=test_dict)
plt.show()