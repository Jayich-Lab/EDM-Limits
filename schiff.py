import numpy as _np
import scipy.constants as _c


class MeasurementSettings:
    """Settings to determine a frequency measurement's sensitivity.

    This class defines the properties of a measurement of a frequency
    that is approporiate for a measurement with uncorrelated trapped
    particles (particle_number).  A down_time per measurement is included as well
    as an experimental efficiency factor (efficiency) which defines the contrast
    (i.e., how high the Rabi/Ramsey peak is).

    Attributes:
        particle_number: uncorrelated particles per measurement (1)
        efficiency: experiment efficiency factor (0.99)
        measurement_time: total measurement time in seconds (10 days in units of seconds)
        dead_time: dead time per measurement (30 ms)
        coherence_time: spin precession time (100 s)
    """

    def __init__(
        self,
        particle_number=1,
        efficiency=0.99,
        measurement_time=864000,
        down_time=30e-3,
        coherence_time=100,
    ):
        self.particle_number = particle_number
        self.efficiency = efficiency
        self.measurement_time = measurement_time
        self.down_time = down_time
        self.coherence_time = coherence_time


def frequency_sensitivity_Hz(measurement_settings):
    """Returns the frequency sensitivity of a measurement in Hertz.

    For trapped, uncorrelated particles.

    Args:
        measurement_settings: MeasurementSettings instance

    Returns:
        float, measurement frequency sensitivity in Hz.
    """
    ms = measurement_settings
    T_total = ms.measurement_time
    T_down = ms.down_time
    tau = ms.coherence_time
    beta = ms.efficiency
    N = ms.particle_number
    prefactor = 2 * _np.pi * tau * beta

    denominator = prefactor * _np.sqrt(N * T_total / (tau + T_down))

    return 1 / denominator


class Molecule:
    """Molecule sensitivity properties to beyond the standard model physics.

    This holds properties such as the molecular enhancement (W_S) and
    nuclear Schiff moment (S).  The a_0, a_1, a_2 values all come from
    Engel2013 and the defaults are set to the radium-225 values.  The W_S value 
    for RaSH+ is unpublished work from Anastia Borschevsky's group.

    The nucleus_E_field_alignment factor is likely correct for RaSH+ and RaOCH3+,
    but thought needs to be given for other molecules, e.g. RaOH+.

    Attributes:
        W_S: molecular enhancment factor in atomic units (45000)
        K_S: Schiff moment/theta enhancement factor given in units of e fm^3 (1.0)
        a_0: isoscalar sensitivity, unitless (-1.5)
        a_1: isovector sensitivity, unitless (6.0)
        a_0: isotensor sensitivity, unitless (-4.0)
        molecule_name: str, name of molecule, default(None)
        nucleus_E_field_alignment: alignment between the nuclear spin
            and the molecule orientation (0.25)
    """

    def __init__(self, W_S=45000, K_S=1.0, a_0=-1.5, a_1=6.0, a_2=-4.0):
        self.W_S = W_S
        self.K_S = K_S
        self.a_0 = a_0
        self.a_1 = a_1
        self.a_2 = a_2
        self.molecule_name = None
        self.nucleus_E_field_alignment = 0.25

    @property
    def schiff_SI(self):
        """Schiff moment in SI units.

        Converts self.K_S to a Schiff moment in SI units.  The
        SI units for the Schiff moment are A*s*m^3, and C*m^3 in
        SI derived units.   When W_S_SI and schiff_SI are multiplied
        you get a value in Joules.  See Yu2021's supplementary material
        for more information.

        Returns:
            float, Schiff moment in SI units.
        """
        fm = 1e-15  # m
        Schiff_au_to_SI = _c.e * fm**3.0
        return Schiff_au_to_SI * self.K_S

    @property
    def W_S_SI(self):
        """Molecular enhancement factor in SI units.

        Converts the self.W_S which is in atomic units to SI units.
        W_S in atomic units is given in e/(4 pi epsilon_0 a_0^4).
        The SI units for W_S are  kg/(A*s^3*m), in SI derived units
        W_S is J/(C*m^3). See schiff_SI's docstring for further discussion.

        Returns:
            float, molecular enhancement in SI units.
        """
        bohr_radius = _c.physical_constants["Bohr radius"][0]  # m
        electric_const = 4.0 * _np.pi * _c.epsilon_0
        W_S_au_to_SI = _c.e / (electric_const * bohr_radius**4.0)
        return W_S_au_to_SI * self.W_S


def theta_QCD_sensitivity(measurement_settings, molecule):
    """A measurement's sensitivity to theta_QCD.

    From the MeasurementSettings and molecule, determine the
    sensitivity to theta_QCD.

    The conversion is done based on values in the supplementary
    material of Yu2021.

    Args:
        measurement_settings: MeasurementSettings instance
        molecule: Molecule instance

    Returns:
        float, theta_QCD sensitivity (a unitless value)
    """
    delta_omega = 2 * _np.pi * frequency_sensitivity_Hz(measurement_settings)
    S = molecule.schiff_SI
    W_S = molecule.W_S_SI

    # The factor of 2 below comes from using two states with opposite
    # Schiff sensitivity.
    opposite_state_enhancement = 2
    orientation_enhancement = 1.0 / (opposite_state_enhancement * molecule.nucleus_E_field_alignment)

    return orientation_enhancement * _c.hbar * delta_omega / (W_S * S)


def schiff_moment_sensitivity(measurement_settings, molecule):
    """A measurement's sensitivity to the nuclear Schiff moment.

    We use the relationship between the Schiff moment and
    Theta_QCD to extract the Schiff moment limit.  See Flambaum2020a
    Eq. 10.

    Args:
        measurement_settings: MeasurementSettings instance
        molecule: Molecule instance

    Returns:
        float, limit on the absolute value of the Schiff moment (e * fm**3)
    """
    theta_limit = theta_QCD_sensitivity(measurement_settings, molecule)

    return molecule.K_S * theta_limit


def _g_pi_NN_to_Schiff_prefactor():
    """Factor to convert from a_0*g_0, a_1*g_1, etc. to a Schiff moment.

    This the is the prefactor in Eq. 4.168 of Engel2013.  This also uses Eq. 3.144
    of Engel2013.

    Notes on m_N:
    The paper does not give a value for m_N, though this is not mass (apparently)
    of a proton or neutron.  We can figure out the mass by plugging in values and solving for that mass using the Graner2016 result (and their calculations).
    It is unclear how Graner settled on this value for m_N.

    Returns:
        float, unitless prefactor
    """
    # Eq. 3.144 of Engel2013
    delta_u_p = 0.746
    delta_u_n = -0.508
    g_A = delta_u_p - delta_u_n

    F_pi = 185  # MeV - pion decay constant
    m_N = 995  # MeV - nucleon mass, note this is larger than the neutron mass.

    # Eq. 4.168  of Engel2013
    return 2.0 * m_N * g_A / F_pi


def g_0_sensitivity(measurement_settings, molecule):
    """A measurement's sensitivity to g_0.

    g_0 is the isoscalar pion nucleon nucleon CP violating parameter.

    Args:
        measurement_settings: MeasurementSettings instance
        molecule: Molecule instance

    Returns:
        float, g_0 sensitivity (a unitless value)
    """
    schiff_limit = schiff_moment_sensitivity(measurement_settings, molecule)
    prefactor = _g_pi_NN_to_Schiff_prefactor()
    return _np.abs(schiff_limit / (prefactor * molecule.a_0))


def g_1_sensitivity(measurement_settings, molecule):
    """Returns a measurement's sensitivity to g_1.

    g_1 is the isovector pion nucleon nucleon CP violating parameter.

    Args:
        measurement_settings: MeasurementSettings instance
        molecule: Molecule instance

    Returns:
        float, g_1 sensitivity (a unitless value)
    """
    schiff_limit = schiff_moment_sensitivity(measurement_settings, molecule)
    prefactor = _g_pi_NN_to_Schiff_prefactor()
    return schiff_limit / (prefactor * molecule.a_1)


def g_2_sensitivity(measurement_settings, molecule):
    """A measurement's sensitivity to g_2.

    g_2 is the isotensor pion nucleon nucleon CP violating parameter.

    Args:
        measurement_settings: MeasurementSettings instance
        molecule: Molecule instance

    Returns:
        float, g_2 sensitivity (a unitless value)
    """
    schiff_limit = schiff_moment_sensitivity(measurement_settings, molecule)
    prefactor = _g_pi_NN_to_Schiff_prefactor()
    return _np.abs(schiff_limit / (prefactor * molecule.a_2))


def up_down_quark_difference_chromo_EDM_sensitivity(measurement_settings, molecule):
    """A molecular Schiff moment measurement's sensitivity to quark chromo EDMs.

    The conversion between g_1 and quark chromo EDMs comes from Pospelov2002.
    The abstract of Pospelov2002 gives a value in units of 10^-26 cm, therefore
    when you take this value as a conversion factor you get 2e14 cm^-1, giving
    units of cm for the difference in the up minus down quark chromo EDMs.

    Args:
        measurement_settings: MeasurementSettings instance
        molecule: Molecule instance

    Returns:
        float, d_u - d_d quark chromo EDM sensitivity (units of cm^-1)
    """
    quark_chromo_factor = 2e14
    g_1_limit = g_1_sensitivity(measurement_settings, molecule)
    d_u_minus_d_d = g_1_limit / quark_chromo_factor
    return d_u_minus_d_d


def radium_225_EDM_sensitivity(measurement_settings, molecule):
    """Limit on the atomic EDM of Ra-225 from a molecule measurement in e * cm.

    Uses Eq. 16 in Dzuba2002a.

    Args:
        measurement_settings: MeasurementSettings instance
        molecule: Molecule instance

    Returns:
        float, radium-225 EDM sensitivity (units of e*cm)
    """
    schiff_limit = schiff_moment_sensitivity(measurement_settings, molecule)
    return _np.abs(-8.5e-17 * schiff_limit)


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
