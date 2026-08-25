package com.polartx.app

import android.annotation.SuppressLint
import android.app.Activity
import android.os.Bundle
import android.system.Os
import android.webkit.JavascriptInterface
import android.webkit.WebView
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import java.util.concurrent.Executors

/**
 * A WebView shell around polartx.appbridge.
 *
 * This file is deliberately inert: every feature is a Python function plus a
 * render in app.js, and nothing here should need to change again.  If a
 * change seems to require touching Kotlin, the logic almost certainly wanted
 * to be in Python, where the 238-test suite is.
 *
 * All Python runs on ONE background thread.  Two reasons, and the second is
 * the one that is easy to miss:
 *
 *  - polartx is plain numpy/scipy code with no locking, so a single lane
 *    makes "one run at a time" a property of the app rather than a
 *    discipline every caller has to maintain.
 *  - the first call pays the import of numpy + scipy + matplotlib, several
 *    seconds on a phone.  Serialising means paying it once; the page shows
 *    its boot card until that first reply lands, or the launch looks like a
 *    hang.
 */
class MainActivity : Activity() {

    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var web: WebView
    private var bridge: PyObject? = null

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!Python.isStarted()) {
            // matplotlib builds a font cache on first import and dies on a
            // read-only config dir.  This has to happen BEFORE Python.start();
            // the symptom otherwise is an app that installs fine and crashes
            // on the first compute, from a device you may not have.
            val mpl = filesDir.resolve("mpl").apply { mkdirs() }
            val cache = filesDir.resolve("cache").apply { mkdirs() }
            Os.setenv("MPLCONFIGDIR", mpl.path, true)
            Os.setenv("XDG_CACHE_HOME", cache.path, true)
            Python.start(AndroidPlatform(this))
        }
        web = WebView(this)
        web.settings.javaScriptEnabled = true
        web.addJavascriptInterface(HostBridge(), "host")
        setContentView(web)
        web.loadUrl("file:///android_asset/www/index.html")
    }

    inner class HostBridge {
        /**
         * Async RPC: returns immediately, the reply lands via onHostReply.
         *
         * @JavascriptInterface methods arrive on a binder thread, so doing
         * the work here would freeze the UI for the length of the call.
         */
        @JavascriptInterface
        fun call(id: String, method: String, argsJson: String) {
            executor.execute {
                val reply = try {
                    val py = bridge ?: Python.getInstance()
                        .getModule("polartx.appbridge").also { bridge = it }
                    py.callAttr("call", method, argsJson).toString()
                } catch (e: Exception) {
                    // the same in-band envelope the Python side uses, so the
                    // page has exactly one error branch
                    JSONObject().put("ok", false)
                        .put("error", e.toString()).toString()
                }
                runOnUiThread {
                    web.evaluateJavascript(
                        "window.onHostReply(${JSONObject.quote(id)}," +
                            "${JSONObject.quote(reply)})", null)
                }
            }
        }
    }
}
