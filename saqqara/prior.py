import numpy as np
import swyft


class SaqqaraPrior:
    def __init__(self, bounds, name=None, parnames=None):
        # TODO: Generalise to non-uniform distributions where bounds needs to be
        # handled differently
        self.name = name
        self.bounds = bounds
        if isinstance(bounds, list):
            self.bounds = np.array(bounds)
        if len(self.bounds.shape) == 0:
            raise ValueError("Bounds must be an (N x 2) array.")
        if len(self.bounds.shape) == 1:
            self.bounds = self.bounds[np.newaxis, ...]
        if not self.bounds.shape[1] == 2:
            raise ValueError("Bounds must be an (N x 2) array.")
        if parnames is not None:
            if len(parnames) != self.bounds.shape[0]:
                raise ValueError(
                    "Bounds and parnames must have the same length."
                )
        self.parnames = parnames
        self.transform_samples = swyft.to_numpy32

    def sample(self, N=None):
        if N is None:
            return self.transform_samples(
                np.random.uniform(
                    self.bounds[:, 0], self.bounds[:, 1], self.bounds.shape[0]
                )
            )
        return self.transform_samples(
            np.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1], (N, self.bounds.shape[0])
            )
        )

    def normalise_sample(self, sample, ranges=None):
        if isinstance(ranges, list):
            ranges = np.array(ranges)
        norm_sample = (sample - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )
        if ranges is not None and len(ranges.shape) != 1:
            if ranges.shape != self.bounds.shape:
                raise ValueError(
                    "Ranges must have the same shape as the bounds."
                )
            return norm_sample * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        elif ranges is not None and len(ranges.shape) == 1:
            return norm_sample * (ranges[1] - ranges[0]) + ranges[0]
        else:
            return norm_sample


def get_prior(settings):
    if "priors" in settings.keys():
        parnames = list(settings.get("priors", {}).keys())
        bounds = np.array([settings["priors"][p] for p in parnames])
        return SaqqaraPrior(bounds=bounds, parnames=parnames)
    else:
        raise ValueError("No prior specified in settings.")
