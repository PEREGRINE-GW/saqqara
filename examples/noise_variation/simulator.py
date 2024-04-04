import gw_response as gwr
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import interp1d
import saqqara


class LISA_AET(saqqara.SaqqaraSim):
    def __init__(self, settings):
        super().__init__(settings)
        self.Hubble_over_h = 3.24e-18
        self.overall_rescaling = 1e30  # TODO: Fix this to be not hardcoded
        self.model_settings = settings.get(
            "model",
            {
                "name": "noise_variation",
                "fmin": 3e-5,
                "fmax": 5e-1,
                "deltaf": 1e-6,
                "ngrid": 1000,
                "noise_approx": False,
            },
        )
        if self.model_settings.get("name", None) != "noise_variation":
            raise ValueError(
                f"Incorrect model name in config ({self.model_settings.get('name', None)}), check settings are correct."
            )
        self.setup_detector(self.model_settings)
        self.compute_fixed_noise_matrices()
        self.compute_fixed_response()  # TODO: Change name of this function
        self.setup_coarse_graining()
        self.build_model(self.model_settings)

    def setup_detector(self, model_settings):
        self.LISA = gwr.LISA(
            fmin=model_settings["fmin"],
            fmax=model_settings["fmax"],
            res=model_settings["deltaf"],
            which_orbits="analytic",
        )
        self.f_vec = self.LISA.frequency_vec(
            freq_pts=int(
                np.floor((self.LISA.fmax - self.LISA.fmin) / self.LISA.res + 1)
            )
        )
        self.pixel_map = gwr.Pixel(NSIDE=8)
        self.x_vec = self.LISA.x(self.f_vec)
        self.strain_conversion = (
            4 * np.pi**2 * self.f_vec**3 / 3 / self.Hubble_over_h**2
        )
        self.arms_matrix_rescaled = (
            self.LISA.detector_arms(np.array([0.0])) / self.LISA.armlength
        )

    def compute_fixed_noise_matrices(self):
        self.TM_tdi_matrix = jnp.abs(
            self.overall_rescaling
            * gwr.noise_TM_matrix(
                TDI_idx=1,  # AET TODO: Generatlise to arbitrary TDI combination
                frequency=self.f_vec,
                TM_acceleration_parameters=jnp.ones(shape=(1, 6)),
                arms_matrix_rescaled=self.LISA.detector_arms(
                    time_in_years=jnp.array([0.0])
                )
                / self.LISA.armlength,
                x_vector=self.x_vec,
            )
        )[
            0, ...
        ]  # Extract t = 0 component (was shape (1, freqs, 3, 3))
        self.OMS_tdi_matrix = jnp.abs(
            self.overall_rescaling
            * gwr.noise_OMS_matrix(
                TDI_idx=1,  # AET TODO: Generatlise to arbitrary TDI combination
                frequency=self.f_vec,
                OMS_parameters=jnp.ones(shape=(1, 6)),
                arms_matrix_rescaled=self.LISA.detector_arms(
                    time_in_years=jnp.array([0.0])
                )
                / self.LISA.armlength,
                x_vector=self.x_vec,
            )
        )[0, ...]
        self.temp_TM_noise = self.overall_rescaling * gwr.LISA_acceleration_noise(
            self.f_vec, acc_param=1.0
        )
        self.temp_OMS_noise = self.overall_rescaling * gwr.LISA_interferometric_noise(
            self.f_vec, inter_param=1.0
        )

    def compute_fixed_response(self):
        # TODO: Generalise to any TDI combination
        response = gwr.Response(det=self.LISA)
        coarse_f_vec = jnp.geomspace(self.f_vec[0], self.f_vec[-1], 1000)
        coarse_x_vec = self.LISA.x(coarse_f_vec)
        response.compute_detector(
            times_in_years=jnp.array([0.0]),
            theta_array=self.pixel_map.theta_pixel,
            phi_array=self.pixel_map.phi_pixel,
            frequency_array=coarse_f_vec,
            TDI="AET",
            polarization="LR",
        )
        AA_interpolator = interp1d(
            np.log(coarse_f_vec),
            np.log(
                response.integrated["AET"]["LL"][0, :, 0, 0] / np.sin(coarse_x_vec) ** 2
            ),
            fill_value="extrapolate",  # TODO: Fix this to manage the lower freq grid limit
        )

        EE_interpolator = interp1d(
            np.log(coarse_f_vec),
            np.log(
                response.integrated["AET"]["LL"][0, :, 1, 1] / np.sin(coarse_x_vec) ** 2
            ),
            fill_value="extrapolate",
        )
        TT_interpolator = interp1d(
            np.log(coarse_f_vec),
            np.log(
                response.integrated["AET"]["LL"][0, :, 2, 2]
                / np.sin(coarse_x_vec) ** 2
                / (1 - np.cos(coarse_x_vec))
            ),
            fill_value="extrapolate",
        )

        self.AA_interpolator = lambda f_vec: np.sin(self.LISA.x(f_vec)) ** 2 * np.exp(
            AA_interpolator(np.log(f_vec))
        )
        self.EE_interpolator = lambda f_vec: np.sin(self.LISA.x(f_vec)) ** 2 * np.exp(
            EE_interpolator(np.log(f_vec))
        )
        self.TT_interpolator = (
            lambda f_vec: np.sin(self.LISA.x(f_vec)) ** 2
            * (1 - np.cos(self.LISA.x(f_vec)))
            * np.exp(TT_interpolator(np.log(f_vec)))
        )
        self.response_AET = np.array(
            [
                self.AA_interpolator(self.f_vec),
                self.EE_interpolator(self.f_vec),
                self.TT_interpolator(self.f_vec),
            ]
        ).T  # AET TODO: Generalise to arbitrary TDI combination, shape = (len(f_vec), 3)
        self.response_matrix = jnp.einsum("ij,jk->ijk", self.response_AET, np.eye(3))

    def setup_coarse_graining(self):
        self.coarse_grained_bins = np.unique(
            np.round(
                np.geomspace(
                    self.f_vec[0], self.f_vec[-1], self.model_settings["ngrid"]
                ),
                decimals=-int(np.ceil(np.log10(self.model_settings["deltaf"]))),
            )
        )
        self.coarse_grained_f = (
            self.coarse_grained_bins[1:] + self.coarse_grained_bins[:-1]
        ) / 2  # TODO: This needs generalising for weighted coarse graining
        self.cg_response_AET = np.array(
            [
                self.AA_interpolator(self.coarse_grained_f),
                self.EE_interpolator(self.coarse_grained_f),
                self.TT_interpolator(self.coarse_grained_f),
            ]
        ).T

    def generate_gaussian(self, std):
        return (np.random.normal(0.0, std) + 1j * np.random.normal(0.0, std)) / np.sqrt(
            2
        )

    def sample_TM(self, size=None):
        if size is not None:
            return self.transform_samples(
                np.random.normal(loc=3.0, scale=0.2, size=size)
            )
        else:
            return self.transform_samples(np.random.normal(loc=3.0, scale=0.2))

    def sample_OMS(self, size=None):
        if size is not None:
            return self.transform_samples(
                np.random.normal(loc=15.0, scale=3.0, size=size)
            )
        else:
            return self.transform_samples(np.random.normal(loc=15.0, scale=3.0))

    def sgwb_template(self, f, z_sgwb):
        conversion = 4 * np.pi**2 * f**3 / 3 / self.Hubble_over_h**2
        return (
            self.overall_rescaling
            * 10 ** z_sgwb[0]
            * (f / np.sqrt(f[0] * f[-1])) ** z_sgwb[1]
            / conversion
        )

    def generate_temp_TM_noise(self, z_noise):
        return z_noise[0] ** 2 * self.temp_TM_noise

    def generate_temp_OMS_noise(self, z_noise):
        return z_noise[1] ** 2 * self.temp_OMS_noise

    def TM_noise_matrix(self, z_noise):
        return z_noise[0] ** 2 * self.TM_tdi_matrix

    def OMS_noise_matrix(self, z_noise):
        return z_noise[1] ** 2 * self.OMS_tdi_matrix

    def generate_temp_sgwb(self, z):
        z_sgwb = z[:-2]
        return self.transform_samples(self.sgwb_template(self.f_vec, z_sgwb))

    def generate_noise_matrix(self, z):
        z_noise = z[-2:]
        return self.TM_noise_matrix(z_noise) + self.OMS_noise_matrix(z_noise)

    def generate_quadratic_signal_data(self, z):
        temp_sgwb = self.generate_temp_sgwb(z)
        quadratic_gaussian_data = (
            np.abs(self.generate_gaussian(np.sqrt(temp_sgwb))) ** 2
        )
        return self.transform_samples(
            jnp.einsum("jkl,j->jkl", self.response_matrix, quadratic_gaussian_data)
        )

    def generate_quadratic_TM_data(self, z):
        z_noise = z[-2:]
        temp_TM_noise = jnp.einsum(
            "i,j->ij", self.generate_temp_TM_noise(z_noise), jnp.ones(6)
        )[
            jnp.newaxis, ...
        ]  # Expand to have time dimension
        nij = self.generate_gaussian(np.sqrt(temp_TM_noise))

        t_retarded_factor = gwr.utils.arm_length_exponential(
            self.arms_matrix_rescaled, self.x_vec
        )
        Dijnji = t_retarded_factor * jnp.roll(nij, 3, axis=-1)
        TM_linear_single_link_data = nij + Dijnji

        linear_TM_noise_data = gwr.tdi.build_tdi(
            1,
            TM_linear_single_link_data[..., jnp.newaxis],
            self.arms_matrix_rescaled,
            self.x_vec,  # Expand single link data to have pixel dimension
        )[0, ..., 0]
        return jnp.einsum(
            "...i,...j->...ij",
            linear_TM_noise_data,
            jnp.conj(linear_TM_noise_data),
        )

    def generate_quadratic_OMS_data(self, z):
        z_noise = z[-2:]
        temp_OMS_noise = jnp.einsum(
            "i,j->ij", self.generate_temp_OMS_noise(z_noise), jnp.ones(6)
        )[
            jnp.newaxis, ...
        ]  # Expand to have time dimension
        OMS_linear_single_link_data = self.generate_gaussian(np.sqrt(temp_OMS_noise))

        linear_OMS_noise_data = gwr.tdi.build_tdi(
            1,
            OMS_linear_single_link_data[..., jnp.newaxis],
            self.arms_matrix_rescaled,
            self.x_vec,
        )[0, ..., 0]
        return jnp.einsum(
            "...i,...j->...ij",
            linear_OMS_noise_data,
            jnp.conj(linear_OMS_noise_data),
        )

    def generate_quadratic_noise_data(self, z):
        # TODO: Check with MP whether this approximation actually works here
        noise_matrix = self.generate_noise_matrix(z)
        quadratic_gaussian_data = (
            np.abs(self.generate_gaussian(np.sqrt(noise_matrix))) ** 2
        )
        return self.transform_samples(quadratic_gaussian_data)

    def generate_coarse_grained_data(self, quadratic_data_AET):
        # NOTE: The coarse-grained data is pre-divided by the response matrix
        # to go to strain units
        out = np.zeros(shape=(len(self.coarse_grained_f) - 1, 3))

        for i in range(len(self.coarse_grained_f) - 1):
            mask = (self.f_vec >= self.coarse_grained_f[i]) & (
                self.f_vec < self.coarse_grained_f[i + 1]
            )
            for j in range(3):
                out[i, j] = np.mean(
                    quadratic_data_AET[mask, j] / self.response_AET[mask, j]
                )
        return self.transform_samples(out)

    def build_model(self, model_settings):
        z = self.graph.nodes["z"]
        quadratic_signal_AET = self.graph.node(
            "quadratic_signal_AET", self.generate_quadratic_signal_data, z
        )
        if model_settings["noise_approx"]:
            quadratic_noise_AET = self.graph.node(
                "quadratic_noise_AET", self.generate_quadratic_noise_data, z
            )
        else:
            quadratic_TM_noise_AET = self.graph.node(
                "quadratic_TM_AET", self.generate_quadratic_TM_data, z
            )
            quadratic_OMS_noise_AET = self.graph.node(
                "quadratic_OMS_AET", self.generate_quadratic_OMS_data, z
            )
            quadratic_noise_AET = self.graph.node(
                "quadratic_noise_AET",
                lambda TM, OMS: TM + OMS,
                quadratic_TM_noise_AET,
                quadratic_OMS_noise_AET,
            )
        quadratic_data_AET = self.graph.node(
            "quadratic_data_AET",
            lambda signal, noise: np.diagonal(signal + noise, axis1=-2, axis2=-1),
            quadratic_signal_AET,
            quadratic_noise_AET,
        )
        coarse_grained_data = self.graph.node(
            "coarse_grained_data",
            self.generate_coarse_grained_data,
            quadratic_data_AET,
        )
