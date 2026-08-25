plugins {
    id("com.android.application") version "8.1.4" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
    // Chaquopy 15.x is the newest line whose package repository carries scipy
    // for Python 3.10 — and 3.10 is the newest Python that has an Android
    // scipy wheel at all (chaquo/chaquopy#1237).  Bumping this without
    // re-running the feasibility gate trades scipy away for a newer
    // interpreter, and scipy is not optional here.
    //
    // The version set moves together: this plugin fixes the Python version,
    // which fixes buildPython and CI's setup-python, which fixes the cp310
    // tag on the compiled wheels.  Change one and re-check all four.
    id("com.chaquo.python") version "15.0.1" apply false
}
