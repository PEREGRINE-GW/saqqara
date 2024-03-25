import unittest
import saqqara
import os

import numpy as np
import swyft

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestSimulator(unittest.TestCase):
    def test_simulator(self):
        config_path = (
            os.path.join(os.path.dirname(saqqara.__file__), "defaults")
            + "/default_config.yaml"
        )
        config = saqqara.load_settings(config_path)
        settings = saqqara.get_settings(config)
        saqqara_sim = saqqara.SaqqaraSim(settings)
        prior_sample = saqqara_sim.sample_prior(10000)
        self.assertEqual(prior_sample.shape, (10000, 4))
        self.assertTrue(
            np.all(
                saqqara_sim.prior.bounds
                - np.array(
                    [[-20.0, -5.0], [-5.0, 5.0], [0.0, 6.0], [0.0, 30.0]]
                )
                == 0.0
            )
        )
        print(
            np.all(
                saqqara_sim.prior.bounds
                - np.array(
                    [[-20.0, -5.0], [-5.0, 5.0], [0.0, 6.0], [0.0, 30.0]]
                )
                == 0.0
            )
        )
        self.assertListEqual(
            saqqara_sim.prior.parnames, ["amp", "tilt", "TM", "OMS"]
        )
        self.assertEqual(saqqara_sim.prior.name, "prior")
        self.assertEqual(saqqara_sim.prior.transform_samples, swyft.to_numpy32)
        self.assertTrue("z" in saqqara_sim.graph.nodes)
        shapes, dtypes = saqqara_sim.get_shapes_and_dtypes()
        self.assertEqual(
            shapes,
            {
                "z": (4,),
            },
        )
        self.assertEqual(
            dtypes,
            {"z": np.float32},
        )
        sample = saqqara_sim.sample(10)
        self.assertEqual(sample["z"].shape, (10, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
