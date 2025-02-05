import numpy as _np
from hadronic import EDMLimit


class eEDMLimit(EDMLimit):
    """Measured electron EDM limit."""

    def __init__(self, year, edm_e_cm, ref):
        super().__init__(year, edm_e_cm, ref)

    #    @property
    #   def theta_QCD(self):
    #      """
    #     The conversion from the neutron EDM limit
    #    to a limit on the theta_QCD value is
    #   from PRL 115, 062001 (2015), Eq. 19.
    #  """
    # n_edm_theta_factor = 0.0039  # units of [e*fm*theta_QCD]
    # factor_in_cm = n_edm_theta_factor * (1e2 / 1e15)
    # return self.edm_e_cm / factor_in_cm

    # @property
    # def cEDM(self):
    #   """Gets the bound on chrome-EDM d_d + 0.5 d_u from a neutron EDM bound.

    #  Note this is not d_u - d_d as in the following cEDM bound calculations.

    # See abstract of Pospelov2001 (https://arxiv.org/abs/hep-ph/0010037).
    # Note that we take the prefactor 1+/-0.5 to be 1.

    # Returns:
    #    bound of d_d + 0.5 d_u in cm.
    # """
    # return self.edm_e_cm / 0.55

    @property
    def one_loop_mass_limit(self):
        """Electron EDM limit on new particle masses at the 1 loop level.

        Args:
            eEDM: float, electron EDM in units of e*cm

        Returns:
            float, particle mass in TeV
        """
        return 48 * _np.sqrt(1e-29 / self.edm_e_cm)

    @property
    def two_loop_mass_limit(self):
        """Electron EDM limit on new particle masses at the 2 loop level.

        Args:
            eEDM: float, electron EDM in units of e*cm

        Returns:
            float, particle mass in TeV
        """
        return 2 * _np.sqrt(1e-29 / self.edm_e_cm)
