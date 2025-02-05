import numpy as _np
import schiff as _schiff


class EDMLimit:
    """A measured permament electric dipole moment limit.

    This classes organizes an experimental result.

    Attributes:
        year: float, year of publication
        ref: str, short paper reference in NameYear format
        edm_e_cm: float, reported EDM limit in e*cm units
        system: str, identify specific EDM systems, e.g. "neutron"
    """

    system = None

    def __init__(self, year, edm_e_cm, ref):
        self.year = year
        self.edm_e_cm = edm_e_cm
        self.ref = ref

    @property
    def theta_QCD(self):
        pass

    @property
    def cEDM(self):
        pass

    @property
    def new_particle_mass_from_cEDM(self):
        return chromo_EDM_limits_on_new_particle_mass(self.cEDM)


class NeutronLimit(EDMLimit):
    """Measured Neutron EDM limit."""

    system = "neutron"

    def __init__(self, year, edm_e_cm, ref):
        super().__init__(year, edm_e_cm, ref)

    @property
    def theta_QCD(self):
        """
        The conversion from the neutron EDM limit
        to a limit on the theta_QCD value is
        from PRL 115, 062001 (2015), Eq. 19.
        """
        n_edm_theta_factor = 0.0039  # units of [e*fm*theta_QCD]
        factor_in_cm = n_edm_theta_factor * (1e2 / 1e15)
        return self.edm_e_cm / factor_in_cm

    @property
    def cEDM(self):
        """Gets the bound on chrome-EDM d_d + 0.5 d_u from a neutron EDM bound.

        Note this is not d_u - d_d as in the following cEDM bound calculations.

        See abstract of Pospelov2001 (https://arxiv.org/abs/hep-ph/0010037).
        Note that we take the prefactor 1+/-0.5 to be 1.


        Returns:
            bound of d_d + 0.5 d_u in cm.
        """
        return self.edm_e_cm / 0.55


class HgLimit(EDMLimit):
    """Measured Hg-199 EDM limit."""

    system = "Hg"

    def __init__(self, year, edm_e_cm, ref):
        super().__init__(year, edm_e_cm, ref)

    @property
    def theta_QCD(self):
        """
        The conversion from a Hg-199 EDM limit
        to a limit on the theta_QCD value is
        from Graner, et al. PRL 116, 161601 (2016).  The ratio
        of the Hg-199 EDM to the value of theta_QCD
        is used to convert previous Hg-199 EDM
        measurements to their theta_QCD limit.
        """
        graner_ratio = 7.4e-30 / 1.5e-10
        return self.edm_e_cm / graner_ratio

    @property
    def cEDM(self):
        """Gets the bound on chrome-EDM d_u - d_d from a Hg EDM bound."""
        return cEDM_Hg(self.edm_e_cm)


def cEDM_Hg(d_Hg):
    """Gets the bound on chrome-EDM d_u - d_d from a Hg EDM bound.

    See Graner2016 Table III and Eq. 5.

    Args:
        d_Hg: Hg EDM bound in e*cm.

    Returns:
        bound of d_u - d_d in cm.
    """
    factor = 5.7e-27 / 7.4e-30
    return d_Hg * factor


def chromo_EDM_limits_on_new_particle_mass(d_q=1e-27):
    """1 loop level mass limits in TeV from quark chromo EDM limits.
    This function comes from email exchanges Andrew had with Jordy De Vries
    in Fall, 2021.  This allows one to connect hadronic TSV sensitivity
    to sensitivity to new particles through quark chromo EDM limits.
    For unit conversions hbar * c = 1 = 197 MeV fm
    So you can express an EDM limit like:
    10^-14 fm in units of inverse energy as
    10^-14 fm = 10^-14 fm/(hbar c) = 10^-14 fm/(197 MeV fm)
    = 10^-14/(197) MeV^-1
    Args:
        d_q: float, quark chromo EDM limit in cm.
    Returns:
        float, particle mass sensitivity scale in TeV.
    """
    alpha = 1 / 137  # fine structure constant
    m_q = 5  # MeV - very approximate up quark mass

    prefactor = alpha * m_q / _np.pi

    cm_to_fm = 10**15 / 10**2
    d_q_fm = d_q * cm_to_fm

    fm_to_inv_MeV = 1 / 197
    d_q_inv_MeV = d_q_fm * fm_to_inv_MeV

    MeV_to_TeV = (1.0 / 1e-6) * (1e-12 / 1.0)

    return MeV_to_TeV * _np.sqrt(prefactor * 1 / d_q_inv_MeV)


class XeLimit(EDMLimit):
    """Measured Xe EDM limit."""

    system = "Xe"

    def __init__(self, year, edm_e_cm, ref):
        super().__init__(year, edm_e_cm, ref)

    @property
    def theta_QCD(self):
        """
        Rough estimate from Table V of
        https://doi.org/10.1103/PhysRevC.91.035502.

        Note that Eq. 19 and 20 of Flambaum2020a says
        Xe sensitivity ~ Hg, different from
        the 10x smaller sensitivity that is assumed here.
        """
        Hg_factor = 7.4e-30 / 1.5e-10  # PRL 116, 161601 (2016)
        Xe_Hg_factor = 10.0
        return self.edm_e_cm * Xe_Hg_factor / Hg_factor

    @property
    def cEDM(self):
        """Gets the bound on chrome-EDM d_u - d_d from a Xe EDM bound.

        Uses cEDM_Hg and estimation from Table V of Chupp2015, https://doi.org/10.1103/PhysRevC.91.035502.

        Note that Eq. 19 and 20 of Flambaum2020a says Xe sensitivity ~ Hg, different from
        the 10x smaller sensitivity that is assumed here.


        Returns:
            bound on (d_u - d_d) in cm.
        """
        eff_d_Hg = self.edm_e_cm * 10.0
        return cEDM_Hg(eff_d_Hg)


class TlFLimit(EDMLimit):
    """Measured TlF EDM limit."""

    system = "TlF"

    def __init__(self, year, edm_e_cm, ref):
        super().__init__(year, edm_e_cm, ref)

    @property
    def _schiff_limit(self):
        """Returns the Schiff moment limit from the atomic EDM.

        Uses the values for the TlF Schiff moment and EDM
        given in Table VII of Cho1991.

        Returns:
            float, Schiff moment limit in e fm^3
        """
        Cho1991_Schiff_limit = 4e-10  # e fm^3
        Cho1991_TlF_EDM_limit = 2.9e-23  # e cm
        EDM_to_schiff_for_TlF = Cho1991_Schiff_limit / Cho1991_TlF_EDM_limit
        return EDM_to_schiff_for_TlF * self.edm_e_cm

    @property
    def theta_QCD(self):
        theta_factor = 0.027  # Flambaum2020a, Eq. 17
        return self._schiff_limit / theta_factor

    @property
    def cEDM(self):
        """TlF chromo EDM limit.

        See Eq. 18 of Flambaum2020a:
        S(TlF) \approx (12 d_d + 9 d_u) e fm^2.
        This is approximated here as a factor of 10.

        Returns:
            bound on (d_u + d_d) in cm.
        """
        cEDM_factor = 10  # Flambaum2020a, Eq. 18
        return self._schiff_limit / cEDM_factor / 1e13


class RaLimit(EDMLimit):
    """Measured Radium-225 EDM limit."""

    system = "Ra"

    def __init__(self, year, edm_e_cm, ref):
        super().__init__(year, edm_e_cm, ref)

    @property
    def _schiff_limit(self):
        """Returns the Schiff moment limit from the atomic EDM limit.

        Dzuba2002a Eq. 16

        d(^{225}Ra) = -8.5\times 10^{-17}(S/(e fm^3)) e cm
        which translates to
        kappa_s = -8.5e-4.  # fm^-2

        S = d/kappa_s

        We are considering the absolute value of the atomic EDM,
        so we drop the minus sign from kappa_s below.

        Returns:
            float, Schiff moment limit in e fm^3

        """
        kappa_s = 8.5e-4  # fm**-2
        edm_e_fm = self.edm_e_cm * (1e15 / 1e2)  # convert from e*cm to e*fm
        schiff_moment_limit = edm_e_fm / kappa_s
        return schiff_moment_limit

    @property
    def theta_QCD(self):
        """Limit on the theta QCD parameter.

        Flambaum2019 Eq. 11
        S(^{225}Ra) = 1.0 theta_QCD e fm^3

        Returns:
            float
        """
        return self._schiff_limit

    @property
    def _g1(self):
        """Limit on g1

        From Ban2010 Eq. 7
        S = a1*g*g1, assuming a sole source analysis, i.e. g0=g2=0.
        Ban2010 text:
        g = g\piNN ~ 13.5, see also Engel2013, Table 5.
        """
        # Engel2013 Table 13
        a1 = 6.0
        # Ban2010, see also Table 5, Englel2013, one of 2 values of g given.
        g = 13.5
        g1_limit = self._schiff_limit / (a1 * g)
        return g1_limit

    @property
    def cEDM(self):
        """Limits on quark chromo EDMs

        See Pospleov2002


        Returns:
            bound on (d_u - d_d) in cm.
        """
        return self._g1 / (2e14)


class YbLimit(EDMLimit):
    """Measured Yb EDM limit."""

    system = "Yb"

    def __init__(self, year, edm_e_cm, ref):
        super().__init__(year, edm_e_cm, ref)

    @property
    def _schiff_limit(self):
        """Returns the Schiff moment limit from the atomic EDM limit.

        Flambaum2020a Table III

        d(^{171}Yb) = -1.88\times 10^{-17}(S/(e fm^3)) e cm
        which translates to
        kappa_s = -1.88e-4.  # fm^-2

        S = d/kappa_s

        We are considering the absolute value of the atomic EDM,
        so we drop the minus sign from kappa_s below.

        Returns:
            float, Schiff moment limit in e fm^3

        """
        kappa_s = 1.88e-4  # fm**-2
        edm_e_fm = self.edm_e_cm * (1e15 / 1e2)  # convert from e*cm to e*fm
        schiff_moment_limit = edm_e_fm / kappa_s
        return schiff_moment_limit

    @property
    def theta_QCD(self):
        """Limit on the theta QCD parameter.

        Dzuba2007 Eq. 8, Yb and Hg Schiff moment to CP violation operators
        are about the same.

        Flambaum2020a Table IV
        S(^{199}Hg) = 0.005 theta_QCD e fm^3

        Returns:
            float
        """
        theta_QCD_to_Schiff = 0.005
        return self._schiff_limit / theta_QCD_to_Schiff

    @property
    def cEDM(self):
        """Gets the bound on chrome-EDM d_u - d_d from a Yb EDM bound.

        Dzuba2007 Eq. 9, Assumes that Yb EDM sensitivity to CP violation operators
        is 0.6x of the Hg EDM sensitivity.

        Returns:
            bound on (d_u - d_d) in cm.
        """
        d_Yb_equivalent_d_Hg = 0.6
        eff_d_Hg = self.edm_e_cm / d_Yb_equivalent_d_Hg
        return cEDM_Hg(eff_d_Hg)
