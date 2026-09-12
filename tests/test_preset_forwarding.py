"""The forwarding presets' signatures stay honest.

nr_dtc and bt_edr_adpll wrap wifi_dtc / ble_adpll.  Naming the common
knobs explicitly makes the signature self-documenting, but it introduces
a way to drift: a parameter could be renamed upstream, or its default
changed, leaving the wrapper quietly disagreeing with the thing it wraps.
These tests pin both.
"""
import inspect

import pytest

import polartx.presets as P

FORWARDERS = [
    (P.nr_dtc, P.wifi_dtc, {"bw", "qam", "fout", "lo_pn", "scs"}),
    (P.bt_edr_adpll, P.ble_adpll, {"dpsk"}),
]


@pytest.mark.parametrize("child,parent,own", FORWARDERS,
                         ids=lambda v: getattr(v, "__name__", ""))
def test_named_params_exist_on_the_target(child, parent, own):
    """Every explicitly named parameter must actually be accepted by the
    preset being wrapped — otherwise the wrapper raises TypeError only
    when someone finally passes it."""
    cp = set(inspect.signature(child).parameters) - {"kw"} - own
    pp = set(inspect.signature(parent).parameters)
    missing = cp - pp
    assert not missing, f"{child.__name__} names {missing}, absent from " \
                        f"{parent.__name__}"


@pytest.mark.parametrize("child,parent,own", FORWARDERS,
                         ids=lambda v: getattr(v, "__name__", ""))
def test_defaults_match_the_target(child, parent, own):
    """A wrapper that silently changes a default would make the two
    presets disagree for the same nominal configuration."""
    cs = inspect.signature(child).parameters
    ps = inspect.signature(parent).parameters
    for name in set(cs) - {"kw"} - own:
        if name not in ps:
            continue
        cd, pd = cs[name].default, ps[name].default
        if cd is inspect.Parameter.empty or pd is inspect.Parameter.empty:
            continue
        assert cd == pd, (f"{child.__name__}({name}={cd!r}) disagrees with "
                          f"{parent.__name__}({name}={pd!r})")


def test_kw_escape_hatch_still_reaches_the_target():
    """**kw must still forward parameters that are NOT named explicitly."""
    p = P.nr_dtc(bw=100e6, range_ui=1.0, dither=False)   # both via **kw
    assert p.fs_bb > 0
    q = P.bt_edr_adpll("8dpsk", fref=32e6, kdco_est_error=0.01)
    assert q.fs_bb == 32e6


def test_named_params_actually_take_effect():
    """Naming a parameter is only useful if it reaches the chain — assert
    a visible behavioural change, not just that the call succeeds."""
    a = P.nr_dtc(bw=100e6, env_skew_s=0.0)
    b = P.nr_dtc(bw=100e6, env_skew_s=2e-9)
    wf = a.make_waveform(n_symbols=4, seed=0)
    ea = a.tx.run(wf, noise=False, seed=1).evm().db
    eb = b.tx.run(wf, noise=False, seed=1).evm().db
    assert eb > ea + 5.0            # skew must visibly degrade EVM

    # EDR: dp_gain error must degrade DEVM
    c = P.bt_edr_adpll("8dpsk", dp_gain=1.0)
    d = P.bt_edr_adpll("8dpsk", dp_gain=1.08)
    w2 = c.make_waveform(n_syms=300, seed=1)
    dc = c.tx.run(w2, noise=False, seed=1).evm()["devm_pct"]
    dd = d.tx.run(w2, noise=False, seed=1).evm()["devm_pct"]
    assert dd > dc * 1.5
