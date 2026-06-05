from .xray  import get_data as get_xray_data
from .mri   import get_data as get_mri_data
from .skin  import get_data as get_skin_data
from .histo import get_data as get_histo_data


def get_data(cfg: dict):
    loaders = {
        "xray":  get_xray_data,
        "mri":   get_mri_data,
        "skin":  get_skin_data,
        "histo": get_histo_data,
    }
    assert cfg["name"] in loaders, f"Neznámy dataset: {cfg['name']}"
    return loaders[cfg["name"]](cfg)