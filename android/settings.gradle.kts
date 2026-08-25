pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
        // Chaquopy: the Gradle plugin AND the Python package repository pip
        // resolves Android binary wheels from.  numpy/scipy/matplotlib come
        // from here, not from PyPI — PyPI has no Android wheels at all.
        maven("https://chaquo.com/maven")
    }
}
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven("https://chaquo.com/maven")
    }
}
rootProject.name = "polartx"
include(":app")
