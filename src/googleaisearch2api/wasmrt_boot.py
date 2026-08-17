"""JavaScript sources bootstrapped into the QuickJS context (data only).

These strings are injected once after the context is created and never
change at runtime — kept in a separate module so the runtime logic stays
linear and easy to review.
"""

# Browser-ish globals a Bot-detection prelude may expect. Only fills the gaps:
# the AToB intrinsic provides btoa/atob, the native extensions provide
# TextEncoder/TextDecoder and WebCrypto. A degraded JS sha-256-aware crypto
# fallback is intentionally NOT provided here — either the crypto extension
# loads (always shipped) or hashing is done in Python.
_JS_PRELUDE = r"""
(function () {
  if (globalThis.__wasmrt_prelude_done) return;
  globalThis.__wasmrt_prelude_done = true;
  var print_ = globalThis.wasmrt_print || function () {};
  var console = {
    log:   function () { print_(Array.prototype.map.call(arguments, String).join(" ") + "\n"); },
    info:  function () { print_(Array.prototype.map.call(arguments, String).join(" ") + "\n"); },
    error: function () { print_(Array.prototype.map.call(arguments, String).join(" ") + "\n"); },
    warn:  function () { print_(Array.prototype.map.call(arguments, String).join(" ") + "\n"); },
    debug: function () {},
  };
  globalThis.console = console;
  if (!globalThis.queueMicrotask) {
    globalThis.queueMicrotask = function (f) { Promise.resolve().then(f); };
  }
  if (!globalThis.clearTimeout) globalThis.clearTimeout = function () {};
  var timers = new Map();
  var nextTimer = 1;
  globalThis.setTimeout = function (fn, ms) {
    var id = nextTimer++;
    var at = Date.now() + (typeof ms === 'number' && ms > 0 ? ms : 0);
    timers.set(id, at);
    var step = function () {
      if (!timers.has(id)) return;
      if (Date.now() >= timers.get(id)) {
        timers.delete(id);
        try { fn(); } catch (e) { console.error(e && e.stack || e); }
      }
      else Promise.resolve().then(step);
    };
    Promise.resolve().then(step);
    return id;
  };
  globalThis.setInterval = function (fn, ms) {
    var id = nextTimer++;
    var period = (typeof ms === 'number' && ms > 0) ? ms : 0;
    var at = Date.now() + period;
    timers.set(id, at);
    var step = function () {
      if (!timers.has(id)) return;
      if (Date.now() >= timers.get(id)) {
        timers.set(id, Date.now() + period);
        try { fn(); } catch (e) { console.error(e && e.stack || e); }
      }
      Promise.resolve().then(step);
    };
    Promise.resolve().then(step);
    return id;
  };
  globalThis.clearInterval = globalThis.clearTimeout;
  if (!globalThis.TextEncoder) {
    globalThis.TextEncoder = function TextEncoder() {};
    globalThis.TextEncoder.prototype.encode = function (s) {
      var out = [];
      s = String(s);
      for (var i = 0; i < s.length; i++) {
        var c = s.charCodeAt(i);
        if (c < 0x80) { out.push(c); }
        else if (c < 0x800) { out.push(0xC0 | (c >> 6), 0x80 | (c & 0x3F)); }
        else if (c >= 0xD800 && c <= 0xDBFF) { out.push(0xEF, 0xBF, 0xBD); }
        else { out.push(0xE0 | (c >> 12), 0x80 | ((c >> 6) & 0x3F), 0x80 | (c & 0x3F)); }
      }
      return new Uint8Array(out);
    };
  }
  if (!globalThis.atob) {
    var B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    globalThis.atob = function (a) {
      a = String(a).replace(/[^A-Za-z0-9+\/=]/g, '');
      var out = '';
      var pad = (a.charAt(a.length - 1) === '=' ? 1 : 0) + (a.charAt(a.length - 2) === '=' ? 1 : 0);
      for (var i = 0; i < a.length; i += 4) {
        var n = (B64.indexOf(a[i] || '=') << 18) | (B64.indexOf(a[i + 1] || '=') << 12)
              | (B64.indexOf(a[i + 2] || '=') << 6) | B64.indexOf(a[i + 3] || '=');
        out += String.fromCharCode((n >> 16) & 255, (n >> 8) & 255, n & 255);
      }
      return out.slice(0, out.length - pad);
    };
    globalThis.btoa = function (b) {
      var out = '';
      var s = String(b);
      for (var i = 0; i < s.length; i += 3) {
        var c0 = s.charCodeAt(i), c1 = s.charCodeAt(i + 1), c2 = s.charCodeAt(i + 2);
        var n = (c0 << 16) | ((c1 || 0) << 8) | (c2 || 0);
        out += B64.charAt((n >> 18) & 63) + B64.charAt((n >> 12) & 63)
             + (i + 1 < s.length ? B64.charAt((n >> 6) & 63) : '=')
             + (i + 2 < s.length ? B64.charAt(n & 63) : '=');
      }
      return out;
    };
  }
  if (!globalThis.crypto) {
    globalThis.crypto = {
      getRandomValues: function (arr) {
        for (var i = 0; i < arr.length; i++) arr[i] = (Math.random() * 256) | 0;
        return arr;
      },
    };
  }
  if (globalThis.crypto && !globalThis.crypto.subtle) {
    globalThis.crypto.subtle = {
      digest: function (alg) {
        return Promise.reject(new Error(
          'crypto extension unavailable (unsupported algorithm: ' + String(alg) + ')'
        ));
      },
    };
  }
  globalThis.window = globalThis;
  globalThis.self = globalThis;
})();
void 0;
"""

# Installed once after the prelude: routes the completion value of an eval
# through a promise chain (so async results settle), then prints it.
# Strings are printed raw (console.log semantics), everything else as JSON.
_RESULT_CAPTURE = (
    "globalThis.__wasmrt_out = function (v) { "
    "Promise.resolve(v).then("
    "function (r) { if (typeof r === 'string') { wasmrt_print(r + '\\n'); return; } "
    "var o; try { o = JSON.stringify(typeof r === 'undefined' ? null : r); "
    "if (o === undefined) o = String(r); } catch (e) { o = String(r); } "
    "wasmrt_print(o + '\\n'); }, "
    "function (e) { var m = (e && e.message) ? e.message : String((e && e.stack) || e); "
    "wasmrt_print(JSON.stringify({__error: m}) + '\\n'); }"
    "); }; void 0;"
)


def js_string_literal(s: str) -> str:
    """Single-quoted, escape-safe JavaScript string literal of ``s``."""
    return (
        "'"
        + (
            s.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )
        + "'"
    )


PRELUDE_SOURCE = _JS_PRELUDE
RESULT_CAPTURE_SOURCE = _RESULT_CAPTURE
