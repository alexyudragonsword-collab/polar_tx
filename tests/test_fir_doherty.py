"""Mixed-domain 2-tap FIR + digital-Doherty (Borokhovich RFIC 2026)."""
import numpy as np
import pytest

from polartx.dpa.characteristics import efficiency_curve
from polartx.fir import (delay_for_notch, fir_response, notch_offsets,
                         ooc_noise_suppression_db)
from polartx.presets import bench_wifi7_mlo_fir_borokhovich26


# ------------------------------------------------------ FIR notch math
def test_fir_response_notches():
    tau = delay_for_notch(500e6)
    f = np.array([0.0, 500e6, 1000e6, 1500e6])
    h = np.abs(fir_response(f, tau))
    assert h[0] == pytest.approx(2.0)            # +6 dB at DC (signal)
    assert h[1] < 1e-9                           # notch at 500 MHz
    assert h[2] == pytest.approx(2.0, abs=1e-9)  # peak at 1 GHz
    assert h[3] < 1e-9                            # notch at 1.5 GHz


def test_notch_offsets_odd_multiples():
    offs = notch_offsets(delay_for_notch(500e6), 2e9)
    assert np.allclose(offs, [500e6, 1500e6])


# ------------------------------------------------- FIR OOC suppression
@pytest.fixture(scope="module")
def preset():
    return bench_wifi7_mlo_fir_borokhovich26(bw=40e6, notch_offset_hz=500e6)


def test_deterministic_ooc_notch(preset):
    """On the deterministic OOC content (quantization/DPD residual) the
    FIR carves a deep notch at the programmed offset."""
    wf = preset.make_waveform(n_symbols=3, seed=0)
    rf = preset.fir_tx.run(wf, noise=False)
    r1 = preset.single_tx.run(wf, noise=False)
    supp = ooc_noise_suppression_db(rf, r1, (450e6, 550e6))
    assert supp > 15.0                           # measured ~25 dB


def test_notch_is_configurable():
    """Notch offset tracks the requested value (slide 23 configurability)
    — the deterministic suppression peaks in a band around it."""
    wf = bench_wifi7_mlo_fir_borokhovich26(bw=40e6).make_waveform(
        n_symbols=3, seed=0)
    for off in (400e6, 700e6):
        p = bench_wifi7_mlo_fir_borokhovich26(bw=40e6, notch_offset_hz=off)
        s_at = ooc_noise_suppression_db(p.fir_tx.run(wf, noise=False),
                                        p.single_tx.run(wf, noise=False),
                                        (off - 40e6, off + 40e6))
        s_off = ooc_noise_suppression_db(p.fir_tx.run(wf, noise=False),
                                         p.single_tx.run(wf, noise=False),
                                         (off / 2 - 40e6, off / 2 + 40e6))
        assert s_at > s_off + 8.0                # notch is where requested


def test_fir_does_not_degrade_evm(preset):
    """Paper: no EVM degradation with FIR (a receiver equalizes the
    2-tap group delay)."""
    wf = preset.make_waveform(seed=0)
    e_fir = preset.fir_tx.run(wf, noise=True, seed=1).evm().db
    e_one = preset.single_tx.run(wf, noise=True, seed=1).evm(
        equalize="per_tone").db
    assert e_fir < -35.0
    assert e_fir < e_one + 2.0


def test_lands_in_the_published_evm_class(preset):
    """Published: -40.7 dB at 40 MHz.  Pinned as a class (not a loose
    '< -30') because this chain's EVM is set by the CFR clipping residual
    — identical with noise on and off — so an over-aggressive CFR target
    silently drops it several dB below the paper, which a loose bound
    would never catch (it did exactly that at cfr_papr_db=8.0)."""
    wf = preset.make_waveform(seed=0)
    e = preset.fir_tx.run(wf, noise=True, seed=1).evm().db
    assert -47.0 < e < -38.0, f"EVM {e:.1f} dB is outside the -40.7 dB class"


def test_evm_is_cfr_limited_not_noise_limited(preset):
    """The property the class pin rests on: turning the random impairments
    off barely moves the EVM, because clipping distortion dominates."""
    wf = preset.make_waveform(seed=0)
    on = preset.fir_tx.run(wf, noise=True, seed=1).evm().db
    off = preset.fir_tx.run(wf, noise=False, seed=1).evm().db
    assert abs(on - off) < 1.0


# ------------------------------------------------- Doherty efficiency
def test_doherty_double_hump():
    x = np.linspace(0.02, 1.0, 400)
    d = efficiency_curve(("doherty", 0.55, 0.35, 6.0), x)
    s = efficiency_curve(("scpa", 0.55, 0.35), x)
    i6 = np.argmin(np.abs(x - 0.5))              # 6 dB backoff
    assert d[i6] > s[i6] + 0.08                  # Doherty enhancement
    assert d[-1] == pytest.approx(0.35, abs=0.01)   # peak at 0 dB
    assert d[i6] == pytest.approx(0.35, rel=0.05)   # second peak at 6 dB BO
