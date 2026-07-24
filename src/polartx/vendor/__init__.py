"""Vendored (adapted-copy) modules from the sibling repositories.

polartx is self-contained: the phase-path engine comes from
``alexyudragonsword-collab/pll_simulator`` (package ``pllsim``, commit
d7be4712) and the waveform/metrics/PA infrastructure from
``alexyudragonsword-collab/PA_DPD`` (package ``padpd``, commit 44f9ee99).
Each file carries a header naming its origin.  Copies are verbatim except:

- ``pllsim/arch/frac.py``: FracConfig/frac_spur_offsets extracted from
  cppll.py; ``pllsim/arch/adpll.py`` imports them from there (one-line
  patch) so the analog charge-pump blocks are not needed.
- ``pllsim/arch/adpll.py``: ``simulate(..., mod_freq_dp=None, dp_cal=None)``
  extensions — an optional separate trajectory for the direct modulation
  point (range-limited direct-DAC modeling; the FCW path keeps the full
  trajectory), and an optional background two-point gain calibrator
  stepped per cycle from the phase error (Markulic-style sign-sign LMS,
  trace in cal_traces['dp_gain']).  Defaults reproduce upstream behavior.
- ``padpd`` subpackage ``__init__`` files are reduced to the vendored
  subset (no OpenDPD compat / neural / dataset-IO modules).

Everything else in polartx imports these modules exclusively through
``polartx.vendor.pllsim...`` / ``polartx.vendor.padpd...`` so the boundary
stays a single seam; upstream fixes are pulled by re-copying the file and
re-applying the notes above.
"""
