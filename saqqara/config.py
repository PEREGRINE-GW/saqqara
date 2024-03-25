import yaml
import os


def load_settings(config_path, verbose=False):
    if not os.path.exists(config_path):
        raise OSError(f"config file ({config_path}) does not exist")
    with open(config_path, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    if verbose:
        print_settings(settings)
    return settings


def print_settings(settings):
    print("\n")
    for key in settings.keys():
        if type(settings[key]) == dict:
            print(f"{key}:")
            for k in settings[key].keys():
                if type(settings[key][k]) == dict:
                    print(f"  {k}:")
                    for kk in settings[key][k].keys():
                        print(
                            f"    {kk}: {settings[key][k][kk]} ({type(settings[key][k][kk]).__name__})"
                        )
                else:
                    print(
                        f"  {k}: {settings[key][k]} ({type(settings[key][k]).__name__})"
                    )

        else:
            print(f"{key}: {settings[key]} ({type(settings[key]).__name__})")
    print("\n")


def get_settings(settings={}):
    settings_to_return = {}
    default_config = load_settings(
        config_path=os.path.join(os.path.dirname(__file__), "defaults/")
        + "default_config.yaml",
        verbose=False,
    )
    settings_to_return["priors"] = settings.get(
        "priors",
        default_config["priors"],
    )
    settings_to_return["run"] = settings.get(
        "run",
        default_config["run"],
    )
    settings_to_return["simulate"] = settings.get(
        "simulate",
        default_config["simulate"],
    )
    settings_to_return["train"] = settings.get(
        "train",
        default_config["train"],
    )
    settings_to_return["infer"] = settings.get(
        "infer",
        default_config["infer"],
    )
    return settings_to_return
