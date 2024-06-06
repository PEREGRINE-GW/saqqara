import numpy as np
import swyft


class SaqqaraPrior:
    def __init__(self, bounds, distribution=None, name=None, parnames=None):
        self.name = name
        self.bounds = np.array(bounds)
        self.distribution = distribution
        if len(self.bounds.shape) == 0:
            raise ValueError("Bounds must be an (N x 2) array.")
        if len(self.bounds.shape) == 1:
            self.bounds = self.bounds[np.newaxis, ...]
        if not self.bounds.shape[1] == 2:
            raise ValueError("Bounds must be an (N x 2) array.")
        if parnames is not None:
            if len(parnames) != self.bounds.shape[0]:
                raise ValueError("Bounds and parnames must have the same length.")
        self.parnames = parnames
        self.transform_samples = swyft.to_numpy32

    def sample(self, N=None):
        if self.distribution is None:
            if N is None:
                return self.transform_samples(
                    np.random.uniform(
                        self.bounds[:, 0],
                        self.bounds[:, 1],
                        self.bounds.shape[0],
                    )
                )
            return self.transform_samples(
                np.random.uniform(
                    self.bounds[:, 0],
                    self.bounds[:, 1],
                    (N, self.bounds.shape[0]),
                )
            )
        else:
            if N is None:
                N = self.bounds.shape[0]
            samples = [dist.sample(N, lb, ub) for dist, (lb, ub) in zip(self.distribution, self.bounds)]
            return self.transform_samples(np.array(samples))

    def normalise_sample(self, sample, ranges=None):
        if isinstance(ranges, list):
            ranges = np.array(ranges)
        norm_sample = (sample - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )
        if ranges is not None and len(ranges.shape) != 1:
            if ranges.shape != self.bounds.shape:
                raise ValueError("Ranges must have the same shape as the bounds.")
            return norm_sample * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        elif ranges is not None and len(ranges.shape) == 1:
            return norm_sample * (ranges[1] - ranges[0]) + ranges[0]
        else:
            return norm_sample


def get_prior(settings):
    distribution_mapping = {
        'Uniform': UniformPrior,
        'Sine': SinePrior,
        'Cosine': CosinePrior,
        'PowerLaw': PowerLawPrior,
        'Gaussian': GaussianPrior,
        'LogNormal': LogNormalPrior
    }

    if "priors" in settings.keys():
        priors = settings.get("priors", {})
        parnames = list(priors.keys())
        bounds = np.array([priors[p]["bounds"] for p in parnames])
        
        distributions = []
        for p in parnames:
            dist_info = priors[p]
            dist_name = dist_info.get('distribution', 'Uniform')  # Default to 'Uniform' if not specified
            dist_class = distribution_mapping.get(dist_name, UniformPrior)
            dist_params = dist_info.get('params', {})
            distributions.append(dist_class(**dist_params))

        return SaqqaraPrior(bounds=bounds, parnames=parnames, distribution=distributions)
    else:
        raise ValueError("No prior specified in settings.")


# Distribution classes

class UniformPrior:
    def __init__(self, lower_bound=0, upper_bound=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self, n_samples, lower_bound=None, upper_bound=None):
        if lower_bound is None:
            lower_bound = self.lower_bound
        if upper_bound is None:
            upper_bound = self.upper_bound
        return np.random.uniform(lower_bound, upper_bound, n_samples)
    
class SinePrior:
    def __init__(self, lower_bound=0, upper_bound=np.pi):
        self.uniform_prior = UniformPrior()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sine_cdf_inverse(self, u):
        return np.arccos(1 - 2 * u)

    def sample(self, n_samples, lower_bound=None, upper_bound=None):
        if lower_bound is None:
            lower_bound = self.lower_bound
        if upper_bound is None:
            upper_bound = self.upper_bound
        if lower_bound < 0 or upper_bound > np.pi:
            raise ValueError("The SinePrior domain is within 0 and \\(\pi\\).")
        extra_samples = int(n_samples * (upper_bound - lower_bound) / np.pi * 2)
        total_samples = n_samples + extra_samples
        uniform_samples = self.uniform_prior.sample(total_samples)
        sine_samples = self.sine_cdf_inverse(uniform_samples)
        filtered_samples = sine_samples[(sine_samples >= lower_bound) & (sine_samples <= upper_bound)]
        while len(filtered_samples) < n_samples:
            extra_uniform_samples = self.uniform_prior.sample(total_samples)
            extra_sine_samples = self.sine_cdf_inverse(extra_uniform_samples)
            extra_filtered_samples = extra_sine_samples[(extra_sine_samples >= lower_bound) & (extra_sine_samples <= upper_bound)]
            filtered_samples = np.concatenate((filtered_samples, extra_filtered_samples))
        return filtered_samples[:n_samples]

class CosinePrior:
    def __init__(self, lower_bound=-np.pi/2, upper_bound=np.pi/2):
        self.uniform_prior = UniformPrior()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def cosine_cdf_inverse(self, u):
        return np.arccos(1 - 2 * u) - np.pi/2

    def sample(self, n_samples, lower_bound=None, upper_bound=None):
        if lower_bound is None:
            lower_bound = self.lower_bound
        if upper_bound is None:
            upper_bound = self.upper_bound
        if lower_bound < -np.pi/2 or upper_bound > np.pi/2:
            raise ValueError("The CosinePrior domain is within -\\(\pi/2\\) and \\(\pi/2\\).")
        extra_samples = int(n_samples * (upper_bound - lower_bound) / np.pi * 2)
        total_samples = n_samples + extra_samples
        uniform_samples = self.uniform_prior.sample(total_samples)
        cosine_samples = self.cosine_cdf_inverse(uniform_samples)
        filtered_samples = cosine_samples[(cosine_samples >= lower_bound) & (cosine_samples <= upper_bound)]
        while len(filtered_samples) < n_samples:
            extra_uniform_samples = self.uniform_prior.sample(total_samples)
            extra_cosine_samples = self.cosine_cdf_inverse(extra_uniform_samples)
            extra_filtered_samples = extra_cosine_samples[(extra_cosine_samples >= lower_bound) & (extra_cosine_samples <= upper_bound)]
            filtered_samples = np.concatenate((filtered_samples, extra_filtered_samples))
        return filtered_samples[:n_samples]

class PowerLawPrior:
    def __init__(self, alpha, lower_bound=1, upper_bound=10):
        self.alpha = alpha
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def powerlaw_cdf_inverse(self, u):
        return ((self.upper_bound**(self.alpha + 1) - self.lower_bound**(self.alpha + 1)) * u + self.lower_bound**(self.alpha + 1))**(1/(self.alpha + 1))

    def sample(self, n_samples, lower_bound=None, upper_bound=None):
        if lower_bound is None:
            lower_bound = self.lower_bound
        if upper_bound is None:
            upper_bound = self.upper_bound
        if lower_bound < self.lower_bound or upper_bound > self.upper_bound:
            raise ValueError("The PowerLawPrior domain is within the specified bounds.")
        extra_samples = int(n_samples * (upper_bound - lower_bound) / (self.upper_bound - self.lower_bound) * 2)
        total_samples = n_samples + extra_samples
        uniform_samples = np.random.uniform(0, 1, total_samples)
        powerlaw_samples = self.powerlaw_cdf_inverse(uniform_samples)
        filtered_samples = powerlaw_samples[(powerlaw_samples >= lower_bound) & (powerlaw_samples <= upper_bound)]
        while len(filtered_samples) < n_samples:
            extra_uniform_samples = np.random.uniform(0, 1, total_samples)
            extra_powerlaw_samples = self.powerlaw_cdf_inverse(extra_uniform_samples)
            extra_filtered_samples = extra_powerlaw_samples[(extra_powerlaw_samples >= lower_bound) & (extra_powerlaw_samples <= upper_bound)]
            filtered_samples = np.concatenate((filtered_samples, extra_filtered_samples))
        return filtered_samples[:n_samples]

class GaussianPrior:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def sample(self, n_samples, lower_bound=-np.inf, upper_bound=np.inf):
        samples = np.random.normal(self.mean, self.std, n_samples)
        if lower_bound == -np.inf and upper_bound == np.inf:
            return samples
        filtered_samples = samples[(samples >= lower_bound) & (samples <= upper_bound)]
        while len(filtered_samples) < n_samples:
            extra_samples = np.random.normal(self.mean, self.std, n_samples)
            extra_filtered_samples = extra_samples[(extra_samples >= lower_bound) & (extra_samples <= upper_bound)]
            filtered_samples = np.concatenate((filtered_samples, extra_filtered_samples))
        return filtered_samples[:n_samples]

class LogNormalPrior:
    def __init__(self, mean=0, sigma=1):
        self.mean = mean
        self.sigma = sigma

    def sample(self, n_samples, lower_bound=0, upper_bound=np.inf):
        samples = np.random.lognormal(self.mean, self.sigma, n_samples)
        if lower_bound == 0 and upper_bound == np.inf:
            return samples
        filtered_samples = samples[(samples >= lower_bound) & (samples <= upper_bound)]
        while len(filtered_samples) < n_samples:
            extra_samples = np.random.lognormal(self.mean, self.sigma, n_samples)
            extra_filtered_samples = extra_samples[(extra_samples >= lower_bound) & (extra_samples <= upper_bound)]
            filtered_samples = np.concatenate((filtered_samples, extra_filtered_samples))
        return filtered_samples[:n_samples]