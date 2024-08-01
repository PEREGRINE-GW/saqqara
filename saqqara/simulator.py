import swyft
import numpy as np
from .prior import SaqqaraPrior


class SaqqaraSim(swyft.Simulator):
    def __init__(self, settings):
        super().__init__()
        if "priors" in settings.keys():
            self.construct_prior_from_settings(settings)
        else:
            raise Warning("No prior specified in settings, must be specified by hand.")
        # TODO: Figure out how to build graph iteratively
        self.graph = swyft.Graph()
        self.build(self.graph)
        self.transform_samples = swyft.to_numpy32

    def construct_prior_from_settings(self, settings):
        # TODO: Generalise to non-uniform priors
        parnames = list(settings.get("priors", {}).keys())
        bounds = np.array([settings["priors"][p] for p in parnames])
        self.parnames = parnames
        self.bounds = bounds
        self.nparams = bounds.shape[0]
        self.prior = SaqqaraPrior(bounds=bounds, parnames=parnames, name="prior")

    def construct_prior_from_bounds(self, bounds, parnames=None, name=None):
        # TODO: Generalise to non-uniform priors
        self.parnames = parnames
        self.bounds = bounds
        self.prior = SaqqaraPrior(bounds=bounds, parnames=parnames, name=name)

    def sample_prior(self, N=None):
        return self.transform_samples(self.prior.sample(N=N))

    def build(self, graph):
        z = graph.node("z", self.sample_prior, None)
