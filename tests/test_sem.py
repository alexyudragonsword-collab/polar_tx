"""SEM checking uses the conventions instruments and specs actually use."""
import numpy as np
import pytest

from polartx.metrics.masks import default_mask, default_mask_spec
from polartx.metrics.sem import MaskSpec, check_sem, integrate_to_rbw
from polartx.presets import lte20_adpll


def test_rbw_integration_makes_the_verdict_resolution_independent():
    """A per-bin comparison moves with nfft; power integrated in the mask's
    resolution bandwidth does not."""
    p = lte20_adpll()
    wf = p.make_waveform(n_symbols=8, seed=0)
    r = p.tx.run(wf, noise=True, seed=1)
    spec = default_mask_spec(wf)
    got = []
    for nfft in (2048, 8192, 32768):
        f, pdb = r.psd(nfft=nfft)
        got.append(check_sem(f, pdb, spec, channel_bw_hz=wf.bw)["oob_margin_db"])
    assert max(got) - min(got) < 1.0, got


def test_out_of_band_margin_is_the_informative_one():
    """The plain worst-case margin is pinned at ~0 dB by the in-channel
    tangency for ANY passing signal, so it cannot distinguish a chain with
    lots of headroom from one that barely passes. The OOB margin can."""
    p0 = lte20_adpll(env_skew_s=0.0)
    p1 = lte20_adpll(env_skew_s=0.5e-9)
    wf = p0.make_waveform(n_symbols=8, seed=0)
    spec = default_mask_spec(wf)

    from polartx.metrics import check_mask
    margins, oob = [], []
    for p in (p0, p1):
        r = p.tx.run(wf, noise=True, seed=1)
        f, pdb = r.psd(nfft=8192)
        _, m, _ = check_mask(f, pdb, default_mask(wf))
        margins.append(m)
        oob.append(check_sem(f, pdb, spec, channel_bw_hz=wf.bw)["oob_margin_db"])

    # both pass, and the legacy margin cannot tell them apart
    assert abs(margins[0] - margins[1]) < 0.1
    # ...while the OOB margin separates them by >10 dB
    assert oob[0] > oob[1] + 10.0


def test_oob_margin_degrades_monotonically_with_skew():
    wf = lte20_adpll().make_waveform(n_symbols=8, seed=0)
    spec = default_mask_spec(wf)
    prev = None
    for sk in (0.0, 0.5, 1.0, 2.0):
        r = lte20_adpll(env_skew_s=sk * 1e-9).tx.run(wf, noise=True, seed=1)
        f, pdb = r.psd(nfft=8192)
        m = check_sem(f, pdb, spec, channel_bw_hz=wf.bw)["oob_margin_db"]
        if prev is not None:
            assert m < prev
        prev = m
    assert prev < 0.0            # 2 ns must fail out of band


def test_integrate_to_rbw_conserves_power_of_a_flat_floor():
    """N bins of equal power integrated into one window read 10log10(N)
    higher — the defining property of a resolution-bandwidth measurement."""
    f = np.linspace(-1e6, 1e6, 2001)
    psd = np.full_like(f, -80.0)
    df = f[1] - f[0]
    out = integrate_to_rbw(f, psd, rbw_hz=10 * df)
    assert out[len(out) // 2] == pytest.approx(-80.0 + 10 * np.log10(10),
                                               abs=0.2)


def test_provenance_is_machine_readable():
    """Shipped templates must not be mistakable for conformance tables."""
    wf = lte20_adpll().make_waveform(n_symbols=4, seed=0)
    spec = default_mask_spec(wf)
    assert spec.source == "stylized"
    assert spec.is_conformance is False
    assert spec.rbw_hz == 1e6            # the mask's declared measurement BW

    # a spec-sourced table declares itself
    real = MaskSpec(points=spec.points, rbw_hz=1e6, basis="dBm_in_rbw",
                    source="3GPP TS 36.101 Table 6.6.2.1.1-1")
    assert real.is_conformance is True


def test_absolute_mask_requires_a_transmit_power():
    wf = lte20_adpll().make_waveform(n_symbols=4, seed=0)
    spec = MaskSpec(points=default_mask(wf), rbw_hz=1e6,
                    basis="dBm_in_rbw", source="test")
    f = np.linspace(-50e6, 50e6, 1001)
    psd = np.full_like(f, -60.0)
    with pytest.raises(ValueError, match="tx_power_dbm"):
        check_sem(f, psd, spec)
    got = check_sem(f, psd, spec, tx_power_dbm=23.0)
    assert np.isfinite(got["worst_margin_db"])


def test_mask_spec_rejects_malformed_input():
    with pytest.raises(ValueError):
        MaskSpec(points=np.zeros(5))                    # not (N,2)
    with pytest.raises(ValueError):
        MaskSpec(points=np.zeros((3, 2)), basis="nonsense")
