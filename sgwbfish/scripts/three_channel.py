from sklearn import covariance
from sympy import beta
import sgwbfish as sf
import jax
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

signal = sf.PowerLaw(f_pivot=jnp.sqrt(3e-5 * 5e-1))
lisa = sf.LISA()
freq_grid = jnp.geomspace(3e-5, 5e-1, 1_000)
Tc, Nc = 1e6, 100
fish = sf.SGWBFish(Sn=lisa, Sh=signal, freq_grid=freq_grid, Tc=Tc, Nc=Nc)

import matplotlib as mpl

mpl.rcParams["axes.formatter.useoffset"] = False


def get_theta_fixed():
    return jnp.hstack(
        [
            jnp.array([-11.0, 0.0]),
            jnp.hstack([jnp.array([3.0, 15.0]) for i in range(Nc)]),
        ]
    )


def get_theta_varying():
    return jnp.hstack(
        [
            jnp.array([-11.0, 0.0]),
            jnp.hstack(
                [
                    jnp.array(
                        [
                            np.random.normal(3.0, 0.6),
                            np.random.normal(15.0, 3.0),
                        ]
                    )
                    for i in range(Nc)
                ]
            ),
        ]
    )


import time

start_time = time.time()
theta = get_theta_fixed()
covariance_matrix = fish.get_covariance_matrix(theta)
alpha_sensivity = jnp.sqrt(covariance_matrix[0, 0])
beta_sensivity = jnp.sqrt(covariance_matrix[1, 1])
print(alpha_sensivity)
print(beta_sensivity)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

Nsamples = 10
nloops = 500
start_time = time.time()
alpha_sensivities = []
beta_sensivities = []
covariance_matrices = []
for _ in tqdm(range(nloops)):
    theta_sample = jnp.vstack([get_theta_varying() for _ in range(Nsamples)])
    covariance_matrices_sample = fish.get_covariance_matrix_vmap(theta_sample)[
        :, :2, :2
    ]
    alpha_sensivity_sample = jnp.sqrt(covariance_matrices_sample[:, 0, 0])
    beta_sensivity_sample = jnp.sqrt(covariance_matrices_sample[:, 1, 1])
    covariance_matrices.append(covariance_matrices_sample)
    alpha_sensivities.append(alpha_sensivity_sample)
    beta_sensivities.append(beta_sensivity_sample)
alpha_sensivities = jnp.hstack(alpha_sensivities)
beta_sensivities = jnp.hstack(beta_sensivities)
covariance_matrices = jnp.vstack(covariance_matrices)
print(covariance_matrices.shape)
print(alpha_sensivities.shape)
print(beta_sensivities.shape)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

np.save("alpha_sensivities.npy", alpha_sensivities)
np.save("beta_sensivities.npy", beta_sensivities)

if True:
    colors = {
        "blue": "#648FFF",
        "purple": "#785EF0",
        "pink": "#DC267F",
        "orange": "#FE6100",
        "yellow": "#FFB000",
        "black": "#000000",
    }

    def plot_gaussian(theta, mu, sigmasq, color, linewidth):
        plt.plot(
            theta,
            1
            / jnp.sqrt(2 * jnp.pi * sigmasq)
            * jnp.exp(-0.5 * (theta - mu) ** 2 / sigmasq),
            color=color,
            linewidth=linewidth,
        )

    def plot_cov_ellipse(cov_matrix, pos, ax=None, **kwargs):
        from matplotlib.patches import Ellipse

        asq = 0.5 * (cov_matrix[0, 0] + cov_matrix[1, 1]) + 0.5 * np.sqrt(
            (cov_matrix[0, 0] - cov_matrix[1, 1]) ** 2 + 4 * cov_matrix[0, 1] ** 2
        )
        bsq = 0.5 * (cov_matrix[0, 0] + cov_matrix[1, 1]) - 0.5 * np.sqrt(
            (cov_matrix[0, 0] - cov_matrix[1, 1]) ** 2 + 4 * cov_matrix[0, 1] ** 2
        )
        phi = 0.5 * np.arctan2(
            2 * cov_matrix[0, 1], (cov_matrix[0, 0] - cov_matrix[1, 1])
        )
        sig1_ellipse = Ellipse(
            xy=pos,
            width=2 * 1.53 * np.sqrt(asq),
            height=2 * 1.53 * np.sqrt(bsq),
            angle=(180.0 / np.pi) * phi,
            **kwargs,
        )
        sig2_ellipse = Ellipse(
            xy=pos,
            width=2 * 2.48 * np.sqrt(asq),
            height=2 * 2.48 * np.sqrt(bsq),
            angle=(180.0 / np.pi) * phi,
            **kwargs,
        )

        ax.add_patch(sig1_ellipse)
        ax.add_patch(sig2_ellipse)

    fig = plt.figure(figsize=(14, 14))
    ax = plt.subplot(2, 2, 3)
    plt.xlabel(r"$\alpha$", fontsize=28)
    plt.ylabel(r"$\gamma$", fontsize=28)
    import tqdm

    for i in tqdm.tqdm(range(Nsamples * nloops // 10)):
        plot_cov_ellipse(
            covariance_matrices[10 * i, :2, :2],
            [-11.0, 0.0],
            ax=ax,
            facecolor="none",
            linewidth=0.2,
            edgecolor="black",
        )

    plot_cov_ellipse(
        covariance_matrix[:2, :2],
        [-11.0, 0.0],
        ax=ax,
        facecolor="none",
        linewidth=4,
        edgecolor=colors["pink"],
    )
    ax.axvline(
        -11.0 - alpha_sensivity,
        color=colors["blue"],
        linewidth=1.6,
        zorder=-10,
    )
    ax.axvline(
        -11.0 + alpha_sensivity,
        color=colors["blue"],
        linewidth=1.6,
        zorder=-10,
    )
    ax.axvline(
        -11.0 - 2 * alpha_sensivity,
        color=colors["blue"],
        linewidth=0.7,
        zorder=-10,
    )
    ax.axvline(
        -11.0 + 2 * alpha_sensivity,
        color=colors["blue"],
        linewidth=0.7,
        zorder=-10,
    )
    ax.set_xlim(-11.0 - 5.0 * alpha_sensivity, -11.0 + 5.0 * alpha_sensivity)
    ax.axhline(-beta_sensivity, color=colors["blue"], linewidth=1.6, zorder=-10)
    ax.axhline(beta_sensivity, color=colors["blue"], linewidth=1.6, zorder=-10)
    ax.axhline(-2 * beta_sensivity, color=colors["blue"], linewidth=0.7, zorder=-10)
    ax.axhline(2 * beta_sensivity, color=colors["blue"], linewidth=0.7, zorder=-10)
    ax.set_ylim(-5.0 * beta_sensivity, 5.0 * beta_sensivity)
    ax.locator_params(axis="both", nbins=4)

    ax = plt.subplot(2, 2, 1)
    theta_arr_1 = jnp.linspace(
        -11.0 - 5.0 * alpha_sensivity, -11.0 + 5.0 * alpha_sensivity, 1000
    )
    theta_arr_2 = jnp.linspace(-5.0 * beta_sensivity, 5.0 * beta_sensivity, 1000)
    plt.xlabel(r"$\alpha$", fontsize=28)
    for i in range(Nsamples * nloops // 10):
        plot_gaussian(
            theta_arr_1,
            -11.0,
            covariance_matrices[10 * i, 0, 0],
            color="black",
            linewidth=0.4,
        )
    plot_gaussian(
        theta_arr_1,
        -11.0,
        covariance_matrix[0, 0],
        color=colors["pink"],
        linewidth=4,
    )
    ax.axvline(
        -11.0 - alpha_sensivity,
        color=colors["blue"],
        linewidth=1.6,
        zorder=-10,
    )
    ax.axvline(
        -11.0 + alpha_sensivity,
        color=colors["blue"],
        linewidth=1.6,
        zorder=-10,
    )
    ax.axvline(
        -11.0 - 2 * alpha_sensivity,
        color=colors["blue"],
        linewidth=0.7,
        zorder=-10,
    )
    ax.axvline(
        -11.0 + 2 * alpha_sensivity,
        color=colors["blue"],
        linewidth=0.7,
        zorder=-10,
    )
    ax.set_xlim(-11.0 - 5.0 * alpha_sensivity, -11.0 + 5.0 * alpha_sensivity)
    ax.locator_params(axis="both", nbins=4)
    ax.set_yticks([])

    ax = plt.subplot(2, 2, 4)
    plt.xlabel(r"$\gamma$", fontsize=28)
    for i in tqdm.tqdm(range(Nsamples * nloops // 10)):
        plot_gaussian(
            theta_arr_2,
            0.0,
            covariance_matrices[10 * i, 1, 1],
            color="black",
            linewidth=0.7,
        )
    plot_gaussian(
        theta_arr_2,
        0.0,
        covariance_matrix[1, 1],
        color=colors["pink"],
        linewidth=4,
    )
    ax.axvline(-beta_sensivity, color=colors["blue"], linewidth=0.7, zorder=-10)
    ax.axvline(beta_sensivity, color=colors["blue"], linewidth=0.7, zorder=-10)
    ax.axvline(-2 * beta_sensivity, color=colors["blue"], linewidth=0.4, zorder=-10)
    ax.axvline(2 * beta_sensivity, color=colors["blue"], linewidth=0.4, zorder=-10)
    ax.set_xlim(-5.0 * beta_sensivity, 5.0 * beta_sensivity)
    ax.locator_params(axis="both", nbins=4)
    ax.set_yticks([])

    ax = plt.subplot(2, 2, 1)
    ax_inset = ax.inset_axes([1.15, 0.6, 0.7, 0.4])

    ax_inset.set_xlabel(r"$\mathrm{Sensitivity}\,\sigma(\alpha)$", fontsize=22)
    ax_inset.hist(
        alpha_sensivities,
        bins=20,
        density=True,
        alpha=0.7,
        histtype="stepfilled",
        color=colors["black"],
    )
    ax_inset.axvline(alpha_sensivity, lw=4, color=colors["pink"])
    ax_inset.locator_params(axis="both", nbins=5)
    ax_inset.set_yticks([])
    ax_inset.set_xticks([0.0011, 0.0012, 0.0013, 0.0014])

    ax = plt.subplot(2, 2, 4)
    ax_inset = ax.inset_axes([0.3, 1.2, 0.7, 0.4])
    ax_inset.set_xlabel(r"$\mathrm{Sensitivity}\,\sigma(\gamma)$", fontsize=22)
    ax_inset.hist(
        beta_sensivities,
        bins=20,
        density=True,
        alpha=0.7,
        histtype="stepfilled",
        color=colors["black"],
    )
    ax_inset.axvline(beta_sensivity, lw=4, color=colors["pink"])
    ax_inset.locator_params(axis="both", nbins=5)
    ax_inset.set_yticks([])
    ax_inset.set_xticks([0.0040, 0.0045, 0.0050, 0.0055])

    import os

    if not os.path.exists(os.path.dirname(__file__) + "/../plots"):
        os.makedirs(os.path.dirname(__file__) + "/../plots")

    plt.savefig(os.path.dirname(__file__) + "/../plots/minus11_sensitivity.png")
