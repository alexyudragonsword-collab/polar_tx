"""Entry point for the Streamlit web GUI as a onefile Windows exe.

Built by both windows-exe.yml (PyInstaller) and windows-exe-nuitka.yml
(Nuitka).  The gui/ scripts ride along as DATA files — Streamlit runs
them from the extracted/compiled bundle, so their imports are invisible
to the freezer's static analysis; the explicit polartx imports below
pull the whole library into the bundle.  A console window stays open on
purpose (the server log); the default browser opens once the server is
up.
"""
import os
import sys
import threading
import time
import webbrowser

# static imports so the freezer bundles everything the pages use
import matplotlib  # noqa: F401

import polartx.cal.dtc_cal  # noqa: F401
import polartx.cal.polar_dpd  # noqa: F401
import polartx.cal.skew  # noqa: F401
import polartx.chain  # noqa: F401
import polartx.export.rtl  # noqa: F401
import polartx.fir  # noqa: F401
import polartx.guiutil  # noqa: F401
import polartx.measured  # noqa: F401
import polartx.metrics  # noqa: F401
import polartx.montecarlo  # noqa: F401
import polartx.presets  # noqa: F401
import polartx.waveforms.ofdm  # noqa: F401


def _base() -> str:
    if hasattr(sys, "_MEIPASS"):                 # PyInstaller: extracted dir
        return sys._MEIPASS
    if "__compiled__" in globals():              # Nuitka: data next to module
        return os.path.dirname(os.path.abspath(__file__))
    # from source: packaging/launch_web.py -> repo root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    from streamlit.web import cli as stcli

    script = os.path.join(_base(), "gui", "Home.py")
    port = os.environ.get("POLARTX_PORT", "8501")
    url = f"http://localhost:{port}"

    def _open():
        time.sleep(4.0)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if not os.environ.get("POLARTX_NO_BROWSER"):
        threading.Thread(target=_open, daemon=True).start()
    print(f"polartx web GUI: {url}  (close this window to stop the server)")
    sys.argv = ["streamlit", "run", script,
                "--server.port", port,
                "--server.headless", "true",
                "--global.developmentMode", "false",
                "--browser.gatherUsageStats", "false"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
