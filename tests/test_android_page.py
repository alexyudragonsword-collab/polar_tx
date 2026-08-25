"""Run the Android page in a real DOM and click its buttons.

``test_android_parity.py`` checks what text alone can check.  It is not
enough, and this repository has the scar: a build reached a phone with every
Run button doing nothing, because the language pass deleted the controls
from the DOM after boot.  Every static check passed — the ids were in
``index.html``, they just stopped existing a moment later.

So this executes ``app.js`` against ``index.html`` under jsdom, with a stub
bridge, and asserts the clicks actually arrive.  The harness itself is
``tests/android_page_harness.js``.

Locally it skips when node or jsdom is missing.  **CI must not rely on that
skip**: ``.github/workflows/ci.yml`` installs jsdom and runs the harness as
its own step, so a missing dependency there is a hard failure rather than a
green run that tested nothing.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests/android_page_harness.js"


def _node_modules() -> str | None:
    """Where jsdom lives, if it is installed anywhere we can reach."""
    if os.environ.get("NODE_PATH"):
        return os.environ["NODE_PATH"]
    local = ROOT / "node_modules"
    return str(local) if (local / "jsdom").is_dir() else None


def _have_jsdom(env) -> bool:
    probe = subprocess.run(["node", "-e", "require('jsdom')"],
                           capture_output=True, env=env)
    return probe.returncode == 0


def test_the_page_boots_and_every_run_button_reaches_the_bridge():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed (CI runs the harness as its own step)")
    env = dict(os.environ)
    nm = _node_modules()
    if nm:
        env["NODE_PATH"] = nm
    if not _have_jsdom(env):
        pytest.skip("jsdom not installed — `npm install jsdom` "
                    "(CI runs the harness as its own step)")

    r = subprocess.run([node, str(HARNESS)], capture_output=True, text=True,
                       env=env, cwd=str(ROOT))
    assert r.returncode == 0, (
        "the Android page does not work when actually run:\n"
        + r.stdout + r.stderr)
    assert "ok —" in r.stdout, r.stdout


def test_the_harness_covers_every_run_button_the_page_defines():
    """The harness lists the buttons it clicks; a sixth tab added to the
    page with no entry here would be untested and look covered."""
    js = HARNESS.read_text(encoding="utf-8")
    listed = set(json.loads(
        "[" + ",".join(f'"{m}"' for m in
                       __import__("re").findall(r'id: "(run-[a-z]+)"', js)) + "]"))
    html = (ROOT / "android/app/src/main/assets/www/index.html").read_text(
        encoding="utf-8")
    defined = set(__import__("re").findall(r'id="(run-[a-z]+)"', html))
    assert listed == defined, (
        f"harness clicks {sorted(listed)} but the page defines "
        f"{sorted(defined)}")
