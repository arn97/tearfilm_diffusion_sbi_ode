import h5py
import numpy as np


H5_PATH = r"C:\Users\arnab\OneDrive\Desktop\Study material\Research\tearfilm_diffusion_sbi\datasets\trials_many_alldata_new_eqution_halton_multinoise.h5"


def load_dataset(downsample_to: int = 101):
    """
    Loads (I_clean, theta_condition, theta_inference) pairs from the H5 dataset.

    Steps:
    1. Read I_clean of shape [64522, 601]
    2. Uniformly downsample time dimension to `downsample_to`
    3. Extract selected parameters:
         (J_e, b_1, b_2, f_0, h0, ts)
       from key:
         "parameters: J_e, b_1, b_2, h_e, f_0, Pc, phi, h0, ts"
    4. Return:
         I_clean_downsampled: shape [N, downsample_to]
         theta_condition: shape [N, 3] (f_0, h0, ts)
         theta_inference: shape [N, 3] (only J_e, b_1, b_2)
    """

    with h5py.File(H5_PATH, "r") as f:

        # -------------------------
        # 1) Load intensity
        # -------------------------
        I_clean = f["I_clean"][:]  # shape [64522, 601]

        if I_clean.shape[1] != 601:
            raise ValueError("Expected 601 time steps in I_clean.")

        # -------------------------
        # 2) Uniform downsampling
        # -------------------------
        if downsample_to > 601:
            raise ValueError("Cannot upsample beyond 601.")

        indices = np.linspace(0, 600, downsample_to, dtype=int)
        I_clean_ds = I_clean[:, indices]  # shape [N, downsample_to]

        # -------------------------
        # 3) Load parameters
        # -------------------------
        param_key = "parameters: J_e, b_1, b_2, h_e, f_0, Pc, phi, h0, ts"
        parameters = f[param_key][:]  # shape [64522, 9]

        if parameters.shape[1] != 9:
            raise ValueError("Expected 9 parameters.")

        # Extract only required ones
        # Order in file:
        # 0: J_e
        # 1: b_1
        # 2: b_2
        # 3: h_e (not needed)
        # 4: f_0
        # 5: Pc (not needed)
        # 6: phi (not needed)
        # 7: h0
        # 8: ts
        theta_condition = parameters[:, [4, 7, 8]]  # shape [N, 3]
        theta_inference = parameters[:, [0, 1, 2]]  # shape [N, 3]

    return [I_clean_ds.astype(np.float32), theta_condition.astype(np.float32), theta_inference.astype(np.float32)]


if __name__ == "__main__":
    I, theta_condition, theta_inference = load_dataset()
    print("I_clean shape:", I.shape)
    print("theta_condition shape:", theta_condition.shape)
    print("theta_inference shape:", theta_inference.shape)