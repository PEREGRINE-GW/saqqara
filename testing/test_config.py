import unittest
import saqqara
import os

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestConfig(unittest.TestCase):
    def test_load_settings(self):
        config_path = TEST_DATA_PATH + "test_config.yaml"
        test_config = {
            "priors": {
                "amp": [-20.0, -5.0],
                "tilt": [-5.0, 5.0],
                "TM": [0.0, 6.0],
                "OMS": [0.0, 30.0],
            },
            "run": {
                "verbose": False,
                "simulate": False,
                "train": False,
                "infer": False,
            },
            "simulate": {
                "store_name": "data_store",
                "store_size": 100000,
                "chunk_size": 500,
            },
            "train": {
                "trainer_dir": "training_dir",
                "train_fraction": 0.85,
                "train_batch_size": 2048,
                "val_batch_size": 2048,
                "num_workers": 0,
                "device": "cpu",
                "n_devices": 1,
                "min_epochs": 1,
                "max_epochs": 100,
                "early_stopping_patience": 7,
                "learning_rate": 7e-05,
                "num_features": 3,
            },
            "infer": {"prior_samples": 100000, "observation": "None"},
        }
        settings = saqqara.load_settings(config_path, verbose=False)
        self.assertEqual(settings, test_config)
        default_loader = saqqara.get_settings(settings)
        self.assertEqual(default_loader, test_config)
        config_path = (
            os.path.dirname(saqqara.__file__) + "/defaults/default_config.yaml"
        )
        settings = saqqara.load_settings(config_path)
        self.assertEqual(settings, test_config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
