import os

import pyrootutils


def get_path(dir_name: str) -> str:
    root_path = pyrootutils.find_root()
    target_path = os.path.join(root_path, dir_name)
    return target_path


def get_assets_path() -> str:
    return get_path("assets")


def get_model_path() -> str:
    return os.path.join(get_assets_path(), "model")
