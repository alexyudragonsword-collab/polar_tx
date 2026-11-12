"""Vendored (adapted-copy) modules from the sibling repositories.

polartx is self-contained: the phase-path engine comes from
``alexyudragonsword-collab/pll_simulator`` (package ``pllsim``, every file
at commit 5564c64 since 2026-11-12; ``arch/adpll.py`` and ``arch/frac.py``
had been held at d7be4712 until the dp_cal hook moved into the kernel) and
the waveform/metrics/PA infrastructure from
``alexyudragonsword-collab/PA_DPD`` (package ``padpd``, commit 44cbcb3,
on PA_DPD's ``main`` since 2026-10-03, when main was fast-forwarded from
3aa1c24 and made the default branch.  When vendored the commit sat only on
feature branches whose names changed twice, which is why CI fetches the
pin by full SHA rather than by branch; ``pa/thermal.py`` and
``gain_modulation.py`` are at 08b9725, also on ``main``, which fixed
ThermalReferencePA restarting its FIRs every 128-sample block — the other
files are byte-identical between the two commits and keep the older pin).
Each file carries a header naming its origin.  Copies are verbatim except:

- ``pllsim/arch/frac.py``: FracConfig extracted from cppll.py;
  ``pllsim/arch/adpll.py`` imports it from there (one-line patch) so the
  analog charge-pump blocks are not needed.
- ``pllsim/arch/adpll.py``: ``simulate(..., mod_freq_dp=None, dp_cal=None)``
  extensions to the TDC kernel — an optional separate trajectory for the
  direct modulation point (range-limited direct-DAC modeling; the FCW path
  keeps the full trajectory), and an optional background two-point gain
  calibrator stepped every cycle inside the kernel from the phase error
  (Markulic-style sign-sign LMS; a step kernel over a state vector, the
  way upstream's TDC-period and Kdco calibrators run; trace in
  cal_traces['dp_gain']).  Defaults reproduce upstream bit for bit, and the
  compiled and interpreted paths agree bit for bit with the hook in use.
- ``padpd`` subpackage ``__init__`` files are reduced to the vendored
  subset (no OpenDPD compat / neural / DDR-Volterra / reference-PA / I/Q
  front-end modules); ``pa/__init__.py``'s ``load_model`` registry lists
  exactly the vendored model classes.
- ``padpd/pa/presets.py`` is an extraction: ``gmp_opendpd_510`` and
  ``mp_opendpd_500`` only, ``ddr_volterra_default`` dropped with its model.

Everything else in polartx imports these modules exclusively through
``polartx.vendor.pllsim...`` / ``polartx.vendor.padpd...`` so the boundary
stays a single seam; upstream fixes are pulled by re-copying the file and
re-applying the notes above.
"""
