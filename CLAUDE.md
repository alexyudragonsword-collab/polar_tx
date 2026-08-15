# CLAUDE.md

Repo-level guidance for Claude Code sessions in `polar_tx`.

`polartx` is a behavioral simulation library for **digital polar
transmitters**: narrowband (ADPLL two-point PM + digital PA — BLE, BT
EDR/EDGE, LTE ≤20 MHz) and wideband (open-loop DTC PM + digital PA —
WiFi 6/7 ≤320 MHz, 5G NR ≤200 MHz).  It is a physics library, not an
application: almost every bug worth catching here is a *wrong number*,
not a crash.

## Read these first

| File | What it gives you |
| --- | --- |
| `docs/architecture.md` | Module map, the three data-flow objects, the two implementation points that bite, "want to add X → change Y" |
| `CONTRIBUTING.md` | The conventions, each backed by a named test or CI job |
| `README.md` | Results tables and the milestone history |

The five files to read before changing anything substantive:
`chain.py`, `phasemod/` (`base.py` + whichever modulator), `presets.py`,
`guiutil.py`.

## Commands

```bash
pip install -e ".[test,gui,guiqt]"
pytest tests/ -q                       # ~240 tests, ~2 min; must be green
python examples/exNN_*.py              # figures land in examples/out/
streamlit run gui/Home.py              # web GUI
polartx-gui                            # desktop GUI
python tools/vendor_check.py           # vendor drift (CI runs this)
```

`iverilog`, `streamlit` and `PySide6` are optional — the tests that need
them skip cleanly when they are missing, so a green run with skips is
normal locally and NOT normal in CI.

## Git

- Develop, commit and push **only** on `claude/digital-polar-tx-dev-r0c338`.
  Do not push to `main` without being asked each time.
- Do not open a PR unless explicitly asked.
- Commit messages explain **why**, not which files changed.
- GitHub access is through the `mcp__github__*` MCP tools; `gh` is not
  available.

## Rules that are easy to break

**1. Assert physics, not snapshots.** Never hardcode a current output as
the expected value, and never hardcode a count (`len(PRESETS) == 14`
already blocked a legitimate change once). Derive an invariant instead.
See `CONTRIBUTING.md` §1 for the seven assertion shapes already in use.

**2. `src/polartx/vendor/` is an adapted copy, not a fork.** Every file
carries a provenance header; `tools/vendor_check.py` +
CI's `vendor-drift` job verify it against pinned upstream commits.
Prefer wrapping on the `polartx` side over editing a vendored file. If
you must edit one, update `tools/vendor_manifest.json`
(`mode` / `body_sha256` / `reason`).

**3. Both GUIs, or neither.** Compute goes in `src/polartx/guiutil.py`;
the Streamlit (`gui/`) and PySide6 (`src/polartx/guiqt/`) frontends both
wire to it, and both need tests. A new `bench_*` preset must reach
`guiutil.PRESETS` — there is a test that fails otherwise.

**4. Declare the measurement convention.** EVM equalization
(`scalar` vs `per_tone`) is carried by
`PolarResult.evm_equalize_default`, so a plotted constellation uses the
same equalizer as the printed number. Spectral templates carry
`MaskSpec.source`; everything shipped is `"stylized"`.

**5. Never invent a specification limit.** The shipped masks are
engineering templates. This environment cannot reach 3GPP/ETSI/IEEE
documents. If a real conformance table is needed, say so and ask for it
— do not produce plausible-looking numbers.

**6. Engines have validity domains.** `ADPLLTwoPoint` `mode="event"` is
only valid while the per-reference-cycle phase advance is ≪1 UI (BLE
0.8% ok, LTE 29% not). It warns and reports
`ui_per_ref_cycle_p99`. Any new approximation gets the same treatment:
document the domain, detect the violation at runtime. Metrics with
insufficient input raise, they do not return NaN.

**7. `response` mode cannot source a spur claim.** It is linearized. Back
spurious/nonlinear conclusions with `mode="event"` or an analytic
prediction, and pin them with a test.

## Working style that fits this repo

- A physical claim in a docstring or the README should be one you
  actually measured in this session. Numbers here are load-bearing.
- Negative results belong in the code. Several docstrings record things
  that did *not* work (smoothstep phase interpolation losing to linear;
  ILA capping out without `fs_scale_fixed`); keep that habit.
- Prefer vectorized numpy. Per-sample Python loops are for genuinely
  sequential state (the event engine) and should say so.
- The prose docs (`README.md`, `CONTRIBUTING.md`, `docs/`) are in
  Chinese; code, docstrings and commit messages are in English.
