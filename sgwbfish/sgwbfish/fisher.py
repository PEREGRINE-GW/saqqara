import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)


class SGWBFish:
    def __init__(self, Sn, Sh, freq_grid, Tc, Nc):
        self.Sn = Sn
        self.Sh = Sh
        self.freq_grid = freq_grid
        self.Tc = Tc
        self.Nc = Nc
        self.num_signal_params = int(Sh.nparams)
        self.num_noise_params = int(Sn.nparams)

    @partial(jax.jit, static_argnums=(0, 4))
    def integrand_prefactor(self, f, theta_noise, theta_signal, channel="AA"):
        return (
            1.0
            / (
                self.Sn.evaluate(f, theta_noise, channel=channel)
                + self.Sh.evaluate(f, theta_signal)
            )
            ** 2
        )

    @partial(jax.jit, static_argnums=(0,))
    def construct_fisher_block(self, theta_noise, theta_signal):

        fisher_block = 0.0
        for channel in ["AA", "EE", "TT"]:
            r1 = self.Sh.gradient(self.freq_grid, theta_signal)
            r2 = self.Sn.gradient(self.freq_grid, theta_noise, channel=channel)
            gradient_vector = jnp.hstack(
                [
                    r1,
                    r2,
                ]
            )
            outer_product = jnp.einsum(
                "...i,...j->...ij", gradient_vector, gradient_vector
            )
            integrand = jnp.einsum(
                "i,ijk->ijk",
                self.integrand_prefactor(
                    self.freq_grid, theta_noise, theta_signal, channel=channel
                ),
                outer_product,
            )

            # fisher_block = jnp.trapezoid(y=integrand, x=self.freq_grid, axis=0)

            fisher_block += jnp.trapezoid(y=integrand, x=self.freq_grid, axis=0)

        return self.Tc * fisher_block

    def get_fisher_block_batched(self, theta_noise_stacked, theta_signal):
        return jax.vmap(self.construct_fisher_block, in_axes=(0, None))(
            theta_noise_stacked, theta_signal
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_fisher_matrix(self, theta):
        fisher_matrix = self.get_zero_matrix()
        theta_signal = theta[: self.num_signal_params]
        theta_noise = theta[self.num_signal_params :]
        theta_noise_stacked = jnp.reshape(theta_noise, (self.Nc, self.num_noise_params))
        fisher_matrix_blocks = self.get_fisher_block_batched(
            theta_noise_stacked, theta_signal
        )
        for i, fisher_matrix_block in zip(range(self.Nc), fisher_matrix_blocks):
            fisher_matrix = fisher_matrix.at[
                : self.num_signal_params, : self.num_signal_params
            ].add(
                fisher_matrix_block[: self.num_signal_params, : self.num_signal_params]
            )
            fisher_matrix = fisher_matrix.at[
                self.num_signal_params
                + i * self.num_noise_params : self.num_signal_params
                + (i + 1) * self.num_noise_params,
                : self.num_signal_params,
            ].add(
                fisher_matrix_block[
                    self.num_signal_params :,
                    : self.num_signal_params,
                ]
            )

            fisher_matrix = fisher_matrix.at[
                : self.num_signal_params,
                self.num_signal_params
                + i * self.num_noise_params : self.num_signal_params
                + (i + 1) * self.num_noise_params,
            ].add(
                fisher_matrix_block[
                    : self.num_signal_params,
                    self.num_signal_params :,
                ]
            )

            fisher_matrix = fisher_matrix.at[
                self.num_signal_params
                + i * self.num_noise_params : self.num_signal_params
                + (i + 1) * self.num_noise_params,
                self.num_signal_params
                + i * self.num_noise_params : self.num_signal_params
                + (i + 1) * self.num_noise_params,
            ].add(
                fisher_matrix_block[
                    self.num_signal_params :,
                    self.num_signal_params :,
                ]
            )
        return fisher_matrix

    @partial(jax.jit, static_argnums=(0,))
    def get_fisher_matrix_unbatched(self, theta):
        fisher_matrix = self.get_zero_matrix()
        theta_signal = theta[: self.num_signal_params]
        theta_noise = theta[self.num_signal_params :]
        for i in range(self.Nc):
            theta_noise_block = theta_noise[
                i * self.num_noise_params : (i + 1) * self.num_noise_params
            ]
            fisher_matrix_block = self.construct_fisher_block(
                theta_noise_block, theta_signal
            )
            fisher_matrix = fisher_matrix.at[
                : self.num_signal_params, : self.num_signal_params
            ].add(
                fisher_matrix_block[: self.num_signal_params, : self.num_signal_params]
            )
            fisher_matrix = fisher_matrix.at[
                self.num_signal_params
                + i * self.num_noise_params : self.num_signal_params
                + (i + 1) * self.num_noise_params,
                : self.num_signal_params,
            ].add(
                fisher_matrix_block[
                    self.num_signal_params :,
                    : self.num_signal_params,
                ]
            )

            fisher_matrix = fisher_matrix.at[
                : self.num_signal_params,
                self.num_signal_params
                + i * self.num_noise_params : self.num_signal_params
                + (i + 1) * self.num_noise_params,
            ].add(
                fisher_matrix_block[
                    : self.num_signal_params,
                    self.num_signal_params :,
                ]
            )

            fisher_matrix = fisher_matrix.at[
                self.num_signal_params
                + i * self.num_noise_params : self.num_signal_params
                + (i + 1) * self.num_noise_params,
                self.num_signal_params
                + i * self.num_noise_params : self.num_signal_params
                + (i + 1) * self.num_noise_params,
            ].add(
                fisher_matrix_block[
                    self.num_signal_params :,
                    self.num_signal_params :,
                ]
            )
        return fisher_matrix

    @partial(jax.jit, static_argnums=(0,))
    def get_covariance_matrix(self, theta):
        return jnp.linalg.inv(self.get_fisher_matrix(theta))

    def get_covariance_matrix_vmap(self, theta):
        return jax.vmap(self.get_covariance_matrix)(theta)

    @partial(jax.jit, static_argnums=(0,))
    def get_zero_matrix(self):
        return jnp.zeros(
            (
                self.num_signal_params + self.Nc * self.num_noise_params,
                self.num_signal_params + self.Nc * self.num_noise_params,
            )
        )
