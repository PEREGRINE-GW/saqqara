import unittest
import saqqara
import os

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestConfig(unittest.TestCase):
    def test_load_settings(self):
        test_config_path = TEST_DATA_PATH + "test_config.yaml"
        test_settings = saqqara.load_settings(test_config_path, verbose=False)
        default_loader = saqqara.get_settings(test_settings)
        self.assertEqual(default_loader, test_settings)
        config_path = (
            os.path.dirname(saqqara.__file__) + "/defaults/default_config.yaml"
        )
        default_settings = saqqara.load_settings(config_path)
        self.assertEqual(default_settings, test_settings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
