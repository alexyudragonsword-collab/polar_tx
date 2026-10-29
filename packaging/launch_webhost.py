"""Entry script for the frozen Windows exe (PyInstaller and Nuitka).

The exe is ``polartx-gui``: the local host serving the one page, opened in
an Edge/Chrome app window, stopping when that window closes
(``polartx.webhost``).  ``POLARTX_NO_BROWSER=1`` serves without opening a
window, which is how CI smoke-tests the exe over HTTP.

The bridge imports its report modules lazily, inside each method, so that
the phone's first call is cheap.  A freezer follows only static imports, so
they are named here once; ``--collect-submodules polartx`` /
``--include-package=polartx`` in the workflows is the belt to these braces.
"""
import sys

import polartx.appbridge  # noqa: F401  (lazy imports below, made static)
import polartx.guispec  # noqa: F401
import polartx.guiutil  # noqa: F401
from polartx import webhost

if __name__ == "__main__":
    webhost.main(sys.argv[1:])
