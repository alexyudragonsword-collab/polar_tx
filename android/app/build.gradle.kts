plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "com.polartx.app"
    compileSdk = 34                       // Chaquopy's own floor

    defaultConfig {
        applicationId = "com.polartx.app"
        minSdk = 24                       // Chaquopy allows 21; 24 is our choice
        targetSdk = 34
        versionCode = 1
        // tracks the polartx version it bundles; bump together
        versionName = "0.1.0"
        ndk {
            // arm64 for phones, x86_64 for the emulator.  Each ABI carries
            // its own CPython plus numpy/scipy/matplotlib, so every extra
            // one costs tens of MB — drop x86_64 if you never emulate.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

chaquopy {
    defaultConfig {
        // 3.10, not newer: the newest Python for which Chaquopy's repository
        // has a scipy wheel (chaquo/chaquopy#1237).  polartx declares
        // requires-python >=3.10, and the sibling pll_simulator ships the
        // same three binary dependencies on this exact version in a CI build
        // that goes green — which is the evidence this gate rests on, rather
        // than an assumption that it will be fine.
        version = "3.10"
        pip {
            // resolved against Chaquopy's own Android wheel repository;
            // unpinned so pip takes the newest build it has for this Python.
            // matplotlib drags in contourpy/kiwisolver/pillow/fonttools,
            // all of which also need Android wheels — they resolve from the
            // same index in the same pass.
            install("numpy")
            install("scipy")
            install("matplotlib")

            // polartx itself, in one of two forms.
            //
            // WHEELS (compiled build): packaging/android_wheel.py produces
            // one wheel PER ABI, so pip has to choose between them by tag —
            // which it only does from --find-links, never from a path.
            // Installing a wheel by path skips tag matching entirely and
            // puts the arm64 wheel into the x86_64 variant as well; it
            // installs cleanly and fails at import with an ELF header error
            // that points nowhere near here.
            //
            // SDIST (interpreted build): the plain build.  An sdist rather
            // than install("../.."), because a directory install makes the
            // whole repository an input of the pip task — and this Gradle
            // project lives inside that repository, so every AGP task's
            // output would land inside the pip task's input and Gradle 8's
            // validation rejects the build.
            //
            // No --no-index anywhere: it is a GLOBAL pip option and the
            // binary dependencies above come from Chaquopy's index in the
            // same resolve, so scoping the local package that way takes
            // numpy and friends down with it.
            val pysrc = file("pysrc")
            val wheels = pysrc.listFiles { f ->
                f.name.matches(Regex("polartx-.*\\.whl"))
            }?.toList() ?: emptyList()
            val sdists = pysrc.listFiles { f ->
                f.name.matches(Regex("polartx-.*\\.tar\\.gz"))
            }?.toList() ?: emptyList()

            if (wheels.isNotEmpty()) {
                require(sdists.isEmpty()) {
                    "android/app/pysrc/ holds both wheels and an sdist — pip " +
                        "would still have a pure-Python polartx to resolve, " +
                        "so the 'compiled' APK could quietly be the " +
                        "interpreted one.  Remove the sdist first: " +
                        "rm -f android/app/pysrc/polartx-*.tar.gz"
                }
                options("--find-links", pysrc.absolutePath)
                install("polartx")
            } else {
                // fail here, with the command, rather than letting pip say
                // "No matching distribution found for polartx" — which tells
                // nobody what to do about it
                require(sdists.size == 1) {
                    "expected exactly one polartx sdist in android/app/pysrc/ " +
                        "(found ${sdists.size}); from the repo root run: " +
                        "python -m build --sdist --outdir android/app/pysrc ."
                }
                install(sdists[0].absolutePath)
            }
        }
    }
}
