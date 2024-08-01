import jax
import jax.numpy as jnp
import numpy as np
import os
from functools import partial

channels_map = {"AA": 0, "EE": 1, "TT": 2}


class Template:
    def __init__(self, nparams):
        self.nparams = nparams

    def evaluate(self, f, theta):
        raise NotImplementedError

    def gradient(self, f, theta):
        return jax.vmap(jax.grad(self.evaluate, argnums=1), in_axes=(0, None))(f, theta)


class PowerLaw(Template):
    def __init__(self, f_pivot=1e-3):
        super().__init__(nparams=2)
        self.f_pivot = f_pivot

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, f, theta):
        alpha, beta = theta[0], theta[1]
        return 10.0**alpha * (f / self.f_pivot) ** beta


class Amplitude(Template):
    def __init__(self, f_pivot=1e-3):
        super().__init__(nparams=1)
        self.f_pivot = f_pivot

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, f, theta):
        alpha = theta[0]
        return 10.0**alpha * (f / self.f_pivot) ** 0.0


class LISA(Template):
    def __init__(self):
        super().__init__(nparams=2)
        self.LISA_data = self.get_LISA_noise()

    def get_LISA_noise(self):

        try:
            noise = jnp.array(
                np.loadtxt(os.path.dirname(__file__) + "/../data/LISA_strain_noise.txt")
            )

        except FileNotFoundError:
            import gw_response as gwr

            TDI_combination = "AET"
            TDI_idx = gwr.TDI_map[TDI_combination]

            lisa = gwr.LISA()
            freqs = jnp.geomspace(3e-5, 5e-1, 1000)
            tm_noise_matrix = gwr.noise_matrix(
                TDI_idx=TDI_idx,
                frequency=freqs,
                TM_acceleration_parameters=jnp.ones(shape=(1, 6)),
                OMS_parameters=jnp.zeros(shape=(1, 6)),
                arms_matrix_rescaled=lisa.detector_arms(time_in_years=0.0)
                / lisa.armlength,
                x_vector=lisa.x(freqs),
            )

            oms_noise_matrix = gwr.noise_matrix(
                TDI_idx=TDI_idx,
                frequency=freqs,
                TM_acceleration_parameters=jnp.zeros(shape=(1, 6)),
                OMS_parameters=jnp.ones(shape=(1, 6)),
                arms_matrix_rescaled=lisa.detector_arms(time_in_years=0.0)
                / lisa.armlength,
                x_vector=lisa.x(freqs),
            )

            response = gwr.Response(
                ps=gwr.PhysicalConstants(),
                det=gwr.LISA(),
            )

            pixel = gwr.Pixel()

            response.compute_detector(
                times_in_years=jnp.array([0.0]),
                theta_array=pixel.theta_pixel,
                phi_array=pixel.phi_pixel,
                frequency_array=freqs,
                TDI=TDI_combination,
                polarization="LR",
            )

            TM = [
                (
                    tm_noise_matrix[0, :, i, i]
                    / response.integrated["AET"]["LL"][0, :, i, i]
                ).real
                for i in range(3)
            ]

            OMS = [
                (
                    oms_noise_matrix[0, :, i, i]
                    / response.integrated["AET"]["LL"][0, :, i, i]
                ).real
                for i in range(3)
            ]

            noise = jnp.vstack((freqs, TM, OMS)).T
            np.savetxt(
                os.path.dirname(__file__) + "/../data/LISA_strain_noise.txt", noise
            )

        return noise

    @partial(jax.jit, static_argnums=(0, 3))
    def evaluate(self, f, theta, channel="AA"):
        A, P = theta[0], theta[1]
        index = channels_map[channel]

        return (
            4
            * np.pi**2
            * f**3
            * (
                A**2 * jnp.interp(f, self.LISA_data[:, 0], self.LISA_data[:, index + 1])
                + P**2
                * jnp.interp(f, self.LISA_data[:, 0], self.LISA_data[:, index + 4])
            )
            / (3 * (3.24e-18) ** 2)
        )

    def gradient(self, f, theta, channel="AA"):
        to_eval = lambda f, theta: self.evaluate(f, theta, channel=channel)
        return jax.vmap(jax.grad(to_eval, argnums=1), in_axes=(0, None))(f, theta)
