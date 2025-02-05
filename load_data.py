import os
import numpy as _np
from hadronic import (
    EDMLimit,
    NeutronLimit,
    TlFLimit,
    HgLimit,
    XeLimit,
    RaLimit,
    YbLimit,
)
from electron import eEDMLimit


def _get_limit_class(system):
    """Get the hadronic system class to instantiate.

    Args:
        system: str, e.g. "neutron".

    Returns:
        EDMLimit class
    """
    hadronic_map = {
        "neutron": NeutronLimit,
        "TlF": TlFLimit,
        "Hg": HgLimit,
        "Xe": XeLimit,
        "Ra": RaLimit,
        "Yb": YbLimit,
    }

    return hadronic_map.get(system)


def hadronic(system):
    """Load EDM data from a file.

    The system name is used to load data from the correct file
    and instantiate the correct data class.

    Args:
        system: str, e.g. "neutron".

    Returns:
        list of EDMLimit class instances
    """
    file_path = os.path.join("data", "hadronic", system + ".txt")
    data = _np.genfromtxt(file_path, dtype=["<i8", "<f8", "S11"])
    data = _np.atleast_1d(data)  # necessary for data with only one entry
    limit_class = _get_limit_class(system)
    EDMs = []
    for measurement in data:
        year = measurement[0]
        edm_limit = measurement[1]
        ref = measurement[2]
        EDMs.append(limit_class(year=year, edm_e_cm=edm_limit, ref=ref))
    return EDMs


def electron(file_name):
    """Load eEDM data from a file.

    Args:
        file_name: str, file name

    Returns:
        list of EDMLimit class instances
    """
    file_path = os.path.join("data", "electron", f"{file_name}")
    data = _np.genfromtxt(file_path, dtype=["<i8", "<f8", "S11"])
    EDMs = []

    if data.ndim == 0:
        data = _np.expand_dims(data, axis=0)

    for measurement in data:
        year = measurement[0]
        edm_limit = measurement[1]
        # one_loop = one_loop_limit_eEDM_mass_limit(edm_limit)
        # two_loop = two_loop_limit_eEDM_mass_limit(edm_limit)
        ref = measurement[2]
        EDMs.append(eEDMLimit(year=year, edm_e_cm=edm_limit, ref=ref))
    return EDMs
