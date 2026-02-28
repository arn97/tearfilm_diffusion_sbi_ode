# make_workflow.py
import bayesflow as bf


def make_diffnet(
    *,
    widths=(512, 512, 512, 512),
    time_embedding_dim=64,
    diffusion_steps: int = 100,
    integrate_method: str = "two_step_adaptive",
):
    """
    Returns a BayesFlow DiffusionModel with explicit control over the
    number of integration / sampling steps via `integrate_kwargs["steps"]`.

    Notes
    -----
    BayesFlow's DiffusionModel exposes `integrate_kwargs` to configure the
    integration procedure; the default config is:
        {'method': 'two_step_adaptive', 'steps': 'adaptive'}
    so setting `steps` to an int gives you direct control over step count.
    """
    if not isinstance(diffusion_steps, int) or diffusion_steps <= 0:
        raise ValueError("diffusion_steps must be a positive integer.")

    diffnet = bf.networks.DiffusionModel(
        noise_schedule="cosine",
        schedule_kwargs=dict(
            min_log_snr=-15.0,
            max_log_snr=15.0,
            shift=0.0,
            weighting="sigmoid",
        ),
        prediction_type="velocity",
        loss_type="velocity",
        subnet="time_mlp",
        subnet_kwargs=dict(
            widths=widths,
            time_embedding_dim=time_embedding_dim,
            residual=True,
            activation="mish",
            norm="layer",
            dropout=0.0,
            merge="concat",
        ),
        integrate_kwargs=dict(
            method=integrate_method,
            steps=diffusion_steps,
        ),
    )
    return diffnet


def make_workflow(
    *,
    # intrinsic = False,
    widths=(512, 512, 512, 512),
    time_embedding_dim=64,
    diffusion_steps: int = 100,

    # checkpoint_filepath: str = (
        # r"C:\Users\arnab\OneDrive\Desktop\Study material\Research"
        # r"\tearfilm_diffusion_sbi\checkpoints\tearfilm_sbi_diffusion.keras"
    # ),
    # save_best_only: bool = True,
):
    diffnet = make_diffnet(
        widths=widths,
        time_embedding_dim=time_embedding_dim,
        diffusion_steps=diffusion_steps,
    )

    # if intrinsic:
    #     workflow = bf.BasicWorkflow(
    #         simulator=None,
    #         summary_network=None,
    #         inference_network=diffnet,
    #         inference_variables="theta_inference",
    #         inference_conditions=("x", "theta_condition"),
    #         standardize=None,
    #         # checkpoint_filepath=checkpoint_filepath,
    #         # save_best_only=save_best_only,
    #     )
    # else:
    workflow = bf.BasicWorkflow(
        simulator=None,
        summary_network=None,
        inference_network=diffnet,
        inference_variables="c_max",
        inference_conditions="x",
        standardize=None,
        # checkpoint_filepath=checkpoint_filepath,
        # save_best_only=save_best_only,
    )
    return diffnet, workflow


if __name__ == "__main__":
    diffnet, workflow = make_workflow(diffusion_steps=64)
    print("DiffusionModel created.")
    print("integrate_kwargs:", diffnet.integrate_kwargs)
    print("Checkpoint:", workflow.checkpoint_filepath)