import unittest
import saqqara
import os

import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestPrior(unittest.TestCase):
    def test_prior(self):
        prior = saqqara.SaqqaraPrior(bounds=[-1, 1], name="one_param")
        self.assertEqual(prior.bounds.shape, (1, 2))

        sample = (
            prior.sample()
        )  # Call with no arguments, returns a single sample of shape (N,)
        self.assertEqual(sample.shape, (1,))
        sample = prior.sample(
            10
        )  # Call with an integer, returns a sample of shape (N, 10)
        self.assertEqual(sample.shape, (10, 1))

        prior = saqqara.SaqqaraPrior(
            bounds=[[-1, 1], [-1, 1], [-1, 1]],
            name="multi_param",
            parnames=["a", "b", "c"],
        )
        self.assertEqual(prior.bounds.shape, (3, 2))
        self.assertEqual(prior.parnames, ["a", "b", "c"])
        self.assertEqual(prior.name, "multi_param")

        sample = (
            prior.sample()
        )  # Call with no arguments, returns a single sample of shape (3,)
        self.assertEqual(sample.shape, (3,))
        sample = prior.sample(
            10
        )  # Call with an integer, returns a sample of shape (N, 3)
        self.assertEqual(sample.shape, (10, 3))

    def test_prior_sample(self):
        prior = saqqara.SaqqaraPrior(bounds=[-1, 1], name="one_param")
        sample = prior.sample(100)
        self.assertEqual(sample.shape, (100, 1))

        self.assertTrue(np.all(sample >= -1) and np.all(sample <= 1))

        prior = saqqara.SaqqaraPrior(
            bounds=[[-1, 1], [-1, 1], [-1, 1]],
            name="multi_param",
            parnames=["a", "b", "c"],
        )
        sample = prior.sample(100)
        self.assertEqual(sample.shape, (100, 3))
        self.assertTrue(np.all(sample >= -1) and np.all(sample <= 1))

    def test_prior_from_settings(self):
        config_path = (
            os.path.join(os.path.dirname(saqqara.__file__), "defaults")
            + "/default_config.yaml"
        )
        config = saqqara.load_settings(config_path)
        settings = saqqara.get_settings(config)
        prior_from_settings = saqqara.get_prior(settings)
        self.assertEqual(prior_from_settings.bounds.shape, (4, 2))
        self.assertListEqual(prior_from_settings.parnames, ["amp", "tilt", "TM", "OMS"])
        sample = prior_from_settings.sample(100)
        self.assertEqual(sample.shape, (100, 4))
        norm_sample = (sample - prior_from_settings.bounds[:, 0]) / (
            prior_from_settings.bounds[:, 1] - prior_from_settings.bounds[:, 0]
        )
        self.assertTrue(np.all(norm_sample >= 0) and np.all(norm_sample <= 1))
        norm_sample = prior_from_settings.normalise_sample(sample)
        self.assertTrue(np.all(norm_sample >= 0) and np.all(norm_sample <= 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
