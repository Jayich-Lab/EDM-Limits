import pytest

import schiff as _schiff


class Test_schiff:
    def __init__(self):
        self.ms = _schiff.MeasurementSettings()
        self.molecule = _schiff.Molecule()


@pytest.fixture(scope="function")
def Schiff_Wrapper():
    return Test_schiff()


def test_frequency_sensitivity_Hz(Schiff_Wrapper):
    out = _schiff.frequency_sensitivity_Hz(Schiff_Wrapper.ms)

    assert out == pytest.approx(1.73e-5, rel=0, abs=1e-6)


def test_frequency_sensitity_JILA():
    """Molecule number from private communication with Will Cairncross (March 22nd, 2021).

    They achieve a statistical sensitivity of 0.87 mHz.
    """
    N_HfF = 9
    beta_HfF = 0.08

    HfF_hours = 313.8
    T_tot_HfF = 3600 * HfF_hours

    tau_single_shot = 950e-3
    tau_HfF = 700e-3
    T_d_HfF = tau_single_shot - tau_HfF

    ms_HfF_2017 = _schiff.MeasurementSettings(
        particle_number=N_HfF,
        efficiency=beta_HfF,
        measurement_time=T_tot_HfF,
        down_time=T_d_HfF,
        coherence_time=tau_HfF,
    )

    delta_f_HfF = _schiff.frequency_sensitivity_Hz(ms_HfF_2017)

    # Testing the value in mHz.
    assert 0.869 == pytest.approx(1e3 * delta_f_HfF, abs=1e-3)


def test_theta_QCD_sensitivity(Schiff_Wrapper):
    out = _schiff.theta_QCD_sensitivity(Schiff_Wrapper.ms, Schiff_Wrapper.molecule)

    assert out == pytest.approx(1.73e-11, rel=0, abs=1e-12)


def test_g_0_sensitivity(Schiff_Wrapper):
    out = _schiff.g_0_sensitivity(Schiff_Wrapper.ms, Schiff_Wrapper.molecule)

    assert out == pytest.approx(8.6e-13, rel=0, abs=1e-14)


def test_g_1_sensitivity(Schiff_Wrapper):
    out = _schiff.g_1_sensitivity(Schiff_Wrapper.ms, Schiff_Wrapper.molecule)

    assert out == pytest.approx(2.1e-13, rel=0, abs=1e-14)


def test_g_2_sensitivity(Schiff_Wrapper):
    out = _schiff.g_2_sensitivity(Schiff_Wrapper.ms, Schiff_Wrapper.molecule)

    assert out == pytest.approx(3.2e-13, rel=0, abs=1e-14)


def test_up_down_quark_difference_chromo_EDM_sensitivity(Schiff_Wrapper):
    out = _schiff.up_down_quark_difference_chromo_EDM_sensitivity(
        Schiff_Wrapper.ms, Schiff_Wrapper.molecule
    )

    assert out == pytest.approx(1.1e-27, rel=0, abs=1e-28)


def test_radium_225_EDM_sensitivity(Schiff_Wrapper):
    out = _schiff.radium_225_EDM_sensitivity(Schiff_Wrapper.ms, Schiff_Wrapper.molecule)

    assert out == pytest.approx(1.5e-27, rel=0, abs=1e-28)


def test_chromo_EDM_limits_on_new_particle_mass():
    out = _schiff.chromo_EDM_limits_on_new_particle_mass(d_q=1e-27)

    assert out == pytest.approx(15.1, rel=0, abs=1e-1)
