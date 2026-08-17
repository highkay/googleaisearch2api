"""Duck.ai ``x-vqd-hash-1`` challenge solver (core of the browserless Duck engine).

Port of Duck2api's ``internal/duckgo/vqd.go`` flow:

1. The server hands back a ``x-vqd-hash-1`` header whose value is base64 of an
   obfuscated async-IIFE challenge script. ``decode_challenge`` turns it back
   into JS source.
2. ``solve_vqd`` runs the challenge inside the persistent wasmrt QuickJS
   context, but only after installing the Duck2api DOM prelude (verbatim port
   below) — the challenge probes ``document``/``iframe``/``Window``/fake
   WebCrypto, and a result produced without those stubs is rejected server-side.
3. The resolved object is parsed in Python and mutated exactly like Duck2api's
   ``vqdResultMutationScript``: sha256+base64 every ``client_hashes`` entry,
   stamp ``meta`` with origin/stack/duration, keep every other key
   (``server_hashes``, ``signals``, ...) untouched.
4. The result is returned as standard padded base64 of the JSON.

The challenge is dynamic (a new script per server response), so nothing here
hardcodes obfuscated constants or hash values.

Only stdlib at module import time; ``wasmrt`` (and thereby ``wasmtime``) is
imported lazily inside ``solve_vqd``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time

from . import wasmrt_boot

DEFAULT_ORIGIN = "https://duck.ai"

# Duck2api defaultVQDUserAgent (matches the Win32 platform the prelude reports).
CHROME_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)

# Plausible fabricated error stack stamped into meta.stack (Duck2api default).
DEFAULT_STACK = (
    "Error\n"
    "    at l (https://duck.ai/dist/duckai-dist/entry.duckai.c0a8c794abcbc8ee2d3c.js:2:1446307)\n"
    "    at async https://duck.ai/dist/duckai-dist/entry.duckai.c0a8c794abcbc8ee2d3c.js:2:1294181"
)

_SOLVE_TIMEOUT_S = 30.0


def decode_challenge(header_value: str) -> str:
    """Base64-decode an ``x-vqd-hash-1`` header value into JS source.

    Header values are standard base64 but may arrive without padding.
    """
    padded = header_value + "=" * (-len(header_value) % 4)
    return base64.b64decode(padded).decode("utf-8", "replace")


def solve_vqd(
    challenge: str,
    *,
    origin: str = DEFAULT_ORIGIN,
    user_agent: str = CHROME_USER_AGENT,
    stack: str | None = None,
) -> str:
    """Execute a decoded challenge and return the next ``x-vqd-hash-1`` value.

    ``challenge`` is the decoded challenge JS SOURCE (already base64-decoded).
    Returns standard padded base64 of ``JSON.stringify(mutated result)``.
    """
    from . import wasmrt  # lazy: wasmtime is imported at wasmrt module top

    stack_value = stack or DEFAULT_STACK
    try:
        wasmrt.eval_js(
            _DOM_PRELUDE_TEMPLATE.replace(
                "__GO_USER_AGENT_LITERAL__", wasmrt_boot.js_string_literal(user_agent)
            )
            .replace("__GO_ORIGIN_LITERAL__", wasmrt_boot.js_string_literal(origin))
            .replace("__GO_STACK_LITERAL__", wasmrt_boot.js_string_literal(stack_value)),
            timeout_s=_SOLVE_TIMEOUT_S,
        )
        started = time.monotonic()
        output = wasmrt.eval_js(challenge, timeout_s=_SOLVE_TIMEOUT_S)
        duration_ms = int((time.monotonic() - started) * 1000)
    finally:
        # Restore the pre-existing globals the prelude shadows: the runtime is a
        # persistent singleton shared by other consumers of eval_js.
        wasmrt.eval_js(_RESTORE_JS, timeout_s=_SOLVE_TIMEOUT_S)

    try:
        result = json.loads(output)
    except ValueError as exc:
        raise ValueError(f"vqd challenge did not resolve to JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("vqd challenge did not resolve to an object")

    hashes = result.get("client_hashes")
    if not isinstance(hashes, list):
        raise ValueError("vqd challenge result has no client_hashes list")

    result["client_hashes"] = [
        base64.b64encode(hashlib.sha256(str(item).encode("utf-8")).digest()).decode("ascii")
        for item in hashes
    ]

    meta_source = result.get("meta")
    meta = dict(meta_source) if isinstance(meta_source, dict) else {}
    meta["origin"] = origin
    meta["stack"] = stack_value
    meta["duration"] = str(duration_ms)
    result["meta"] = meta

    return base64.b64encode(json.dumps(result, separators=(",", ":")).encode("utf-8")).decode(
        "ascii"
    )


# =============================================================================
# DOM prelude support: everything below is data and glue for the JS above.
# =============================================================================

# Globals the Duck2api prelude (re)defines on globalThis. Snapshot before,
# restore after, so the persistent QuickJS context stays clean for other
# consumers (e.g. the WebCrypto smoke tests) — goja gives Duck2api a fresh VM
# per solve; wasmrt gives us one forever VM, so we emulate that per-solve
# freshness around each solve.
_RESTORABLE_GLOBALS = [
    "window",
    "self",
    "top",
    "document",
    "location",
    "navigator",
    "TextEncoder",
    "Element",
    "HTMLElement",
    "HTMLDivElement",
    "HTMLIFrameElement",
    "HTMLScriptElement",
    "NodeList",
    "__DDG_BE_VERSION__",
    "__DDG_FE_CHAT_HASH__",
    "Window",
    "btoa",
    "atob",
    "getComputedStyle",
    "setTimeout",
    "clearTimeout",
    "setInterval",
    "clearInterval",
    "performance",
    "crypto",
    "screen",
    "history",
    "localStorage",
    "sessionStorage",
    "console",
    "XMLHttpRequest",
    "fetch",
    "URL",
    "URLSearchParams",
    "requestAnimationFrame",
    "cancelAnimationFrame",
    "matchMedia",
    "ResizeObserver",
    "IntersectionObserver",
    "MutationObserver",
    "Image",
]
_NAMES_JS = json.dumps(_RESTORABLE_GLOBALS)

_SNAPSHOT_JS = (
    "(function () { var names = " + _NAMES_JS + ";"
    "var snap = { present: {}, desc: {} };"
    "names.forEach(function (n) {"
    " if (Object.prototype.hasOwnProperty.call(globalThis, n)) {"
    "  snap.present[n] = true;"
    "  snap.desc[n] = Object.getOwnPropertyDescriptor(globalThis, n);"
    " }"
    "});"
    "globalThis.__vqd_snap__ = snap;"
    "})();"
)

_RESTORE_JS = (
    "(function () { var names = " + _NAMES_JS + ";"
    "var snap = globalThis.__vqd_snap__;"
    "if (!snap) return;"
    "Object.keys(snap.desc).forEach(function (n) {"
    " Object.defineProperty(globalThis, n, snap.desc[n]);"
    "});"
    "names.forEach(function (n) { if (!snap.present[n]) delete globalThis[n]; });"
    "globalThis.__vqd_snap__ = undefined;"
    "})();"
)

# allow: SIZE_OK — _DOM_PRELUDE_TEMPLATE is a verbatim port of Duck2api's
# vqdBrowserPrelude (external data, a JS string literal, not Python logic).
#
# Port notes (all changes are OUTSIDE the IIFE body, which is verbatim):
#   * __goUserAgent / __goOrigin / __goStack -> JS var bindings injected before
#     the IIFE (were Goja host bindings in Duck2api).
#   * __goBtoa / __goAtob -> bound to wasmrt's pristine global btoa/atob BEFORE
#     the IIFE redefines those globals. A literal in-body replacement with
#     `btoa(v)`/`atob(v)` would make contentWin__.btoa and the redefined
#     global btoa self-recursive in this PERSISTENT runtime (goja has no
#     pre-existing btoa, so the replacement is only safe there, not here).
#   * __goSha256Base64 -> absent from the prelude body (Duck2api uses it only
#     in the separate mutation script); hashing happens in Python instead.
#   * The fake crypto.subtle.digest returning Promise<ArrayBuffer(32)> is kept
#     EXACTLY as upstream; the challenge result's client_hashes depend on it.
_DOM_PRELUDE_TEMPLATE = (
    _SNAPSHOT_JS
    + "\n"
    + """
if (typeof globalThis.__vqd_pristine_btoa === "undefined") {
  globalThis.__vqd_pristine_btoa = globalThis.btoa;
  globalThis.__vqd_pristine_atob = globalThis.atob;
}
var __goBtoa = globalThis.__vqd_pristine_btoa;
var __goAtob = globalThis.__vqd_pristine_atob;
var __goUserAgent = __GO_USER_AGENT_LITERAL__;
var __goOrigin = __GO_ORIGIN_LITERAL__;
var __goStack = __GO_STACK_LITERAL__;
"""
    + r"""
(function () {
  "use strict";
  var userAgent__ = __goUserAgent;

  // ===== TextEncoder =====
  function TextEncoder__() {}
  TextEncoder__.prototype.encode = function (value) {
    var text = String(value);
    var encoded = encodeURIComponent(text);
    var bytes = [];
    for (var i = 0; i < encoded.length; i++) {
      if (encoded[i] === "%") {
        bytes.push(parseInt(encoded.slice(i + 1, i + 3), 16));
        i += 2;
      } else {
        bytes.push(encoded.charCodeAt(i));
      }
    }
    return new Uint8Array(bytes);
  };

  // ===== Navigator =====
  var navigator__ = {
    userAgent: userAgent__,
    platform: "Win32",
    language: "en-US",
    languages: ["en-US", "en"],
    cookieEnabled: true,
    onLine: true,
    hardwareConcurrency: 4,
    maxTouchPoints: 0,
    vendor: "Google Inc.",
    vendorSub: "",
    productSub: "20030107",
    appCodeName: "Mozilla",
    appName: "Netscape",
    appVersion: userAgent__,
    product: "Gecko",
    doNotTrack: null,
    mimeTypes: {
      length: 0,
      item: function () { return null; },
      namedItem: function () { return null; },
    },
    plugins: {
      length: 0,
      item: function () { return null; },
      namedItem: function () { return null; },
    },
    webdriver: false,
    deviceMemory: 8,
    javaEnabled: function () { return false; },
    getBattery: function () { return Promise.resolve({ level: 1, charging: true }); },
  };

  // ===== DOM constructors =====
  function NodeList__(items) {
    var vals = items || [];
    for (var i = 0; i < vals.length; i++) this[i] = vals[i];
    this.length = vals.length;
  }
  NodeList__.prototype.item = function (i) { return this[i] || null; };
  NodeList__.prototype.forEach = function (fn) {
    for (var i = 0; i < this.length; i++) fn(this[i], i, this);
  };

  function Element__(tagName) {
    var name = String(tagName || "").toUpperCase();
    this.tagName = name;
    this.nodeType = 1;
    this.nodeName = name;
    this.children = [];
    this.parentNode = null;
    this.ownerDocument = null;
    this.attributes = {};
    this.innerHTML = "";
    this.textContent = "";
    this.srcdoc = "";
    this.src = "";
    this.style = {
      cssText: "",
      display: "inline-block",
      getPropertyValue: function (n) {
        return String(n).toLowerCase() === "display" ? this.display || "inline-block" : "";
      },
    };
    this.offsetWidth = 1;
    this.offsetHeight = 1;
    this.scrollHeight = 1;
    this.clientWidth = 1;
    this.clientHeight = 1;
  }
  Element__.prototype.constructor = Element__;
  Element__.prototype.appendChild = function (child) {
    child.parentNode = this;
    child.ownerDocument = this.ownerDocument;
    this.children.push(child);
    return child;
  };
  Element__.prototype.removeChild = function (child) {
    var idx = this.children.indexOf(child);
    if (idx >= 0) { this.children.splice(idx, 1); child.parentNode = null; }
    return child;
  };
  // 内部函数: 匹配属性选择器 meta[http-equiv="..."]
  function matchesAttributeSelector__(el, selector) {
    var re = /^([a-z0-9_-]+)\[([a-z0-9_-]+)=(["']?)([^"'\]]+)\3\]$/i;
    var m = selector.match(re);
    if (!m) return false;
    var tag = m[1].toLowerCase(), attr = m[2], val = m[4];
    if (tag !== "*" && el.tagName && el.tagName.toLowerCase() !== tag) return false;
    return el.getAttribute(attr) === val;
  }

  // 内部函数: 递归收集匹配元素
  function collectMatching__(el, matchFn, results) {
    if (matchFn(el)) results.push(el);
    if (el.children && el.children.length > 0) {
      for (var i = 0; i < el.children.length; i++) {
        collectMatching__(el.children[i], matchFn, results);
      }
    }
  }

  Element__.prototype.querySelectorAll = function (selector) {
    selector = String(selector || "").toLowerCase();
    // 特殊: #jsa
    if (selector === "#jsa" && this.ownerDocument && this.ownerDocument.__jsa__) {
      return new NodeList__([this.ownerDocument.__jsa__]);
    }
    // 特殊: meta[http-equiv="Content-Security-Policy"]
    if (selector.indexOf("meta[") === 0) {
      var results = [];
      if (this.children && this.children.length > 0) {
        for (var i = 0; i < this.children.length; i++) {
          collectMatching__(this.children[i], function(el) {
            return matchesAttributeSelector__(el, selector);
          }, results);
        }
      }
      return new NodeList__(results);
    }
    return new NodeList__([]);
  };
  Element__.prototype.querySelector = function (selector) {
    var list = this.querySelectorAll(selector);
    return list.length > 0 ? list[0] : null;
  };
  Element__.prototype.getAttribute = function (name) {
    return this.attributes[String(name)] || null;
  };
  Element__.prototype.setAttribute = function (name, value) {
    this.attributes[String(name)] = String(value);
  };
  Element__.prototype.getBoundingClientRect = function () {
    return { width: 1, height: 1, top: 0, right: 1, bottom: 1, left: 0 };
  };
  Element__.prototype.addEventListener = function () {};
  Element__.prototype.removeEventListener = function () {};
  Element__.prototype.focus = function () {};
  Element__.prototype.blur = function () {};
  Element__.prototype.cloneNode = function () { return Object.create(this); };

  function HTMLElement__() { Element__.apply(this, arguments); }
  HTMLElement__.prototype = Object.create(Element__.prototype);
  HTMLElement__.prototype.constructor = HTMLElement__;

  function HTMLDivElement__() { HTMLElement__.apply(this, arguments); }
  HTMLDivElement__.prototype = Object.create(HTMLElement__.prototype);
  HTMLDivElement__.prototype.constructor = HTMLDivElement__;

  function HTMLIFrameElement__() { HTMLElement__.apply(this, arguments); }
  HTMLIFrameElement__.prototype = Object.create(HTMLElement__.prototype);
  HTMLIFrameElement__.prototype.constructor = HTMLIFrameElement__;

  function HTMLScriptElement__() { HTMLElement__.apply(this, arguments); }
  HTMLScriptElement__.prototype = Object.create(HTMLElement__.prototype);
  HTMLScriptElement__.prototype.constructor = HTMLScriptElement__;

  function createElement__(tagName) {
    tagName = String(tagName || "").toLowerCase();
    var el;
    if (tagName === "div") el = new HTMLDivElement__();
    else if (tagName === "iframe") el = new HTMLIFrameElement__();
    else if (tagName === "script") el = new HTMLScriptElement__();
    else el = new Element__(tagName);
    Element__.call(el, tagName);
    return el;
  }

  // ===== Document =====
  var docLocation__ = {
    href: "https://duck.ai/",
    origin: __goOrigin,
    protocol: "https:",
    host: "duck.ai",
    hostname: "duck.ai",
    port: "",
    pathname: "/",
    search: "",
    hash: "",
  };

  function makeDocument__() {
    var docEl = new Element__("html");
    var head = new Element__("head");
    var body = new Element__("body");
    docEl.ownerDocument = docEl;
    head.ownerDocument = head;
    body.ownerDocument = body;
    docEl.appendChild(head);
    docEl.appendChild(body);

    var doc = {
      documentElement: docEl,
      head: head,
      body: body,
      cookie: "",
      title: "",
      referrer: "",
      URL: "https://duck.ai/",
      domain: "duck.ai",
      readyState: "complete",
      visibilityState: "visible",
      hidden: false,
      defaultView: null,
      __jsa__: null,
      location: docLocation__,
      createElement: function (tagName) {
        var el = createElement__(tagName);
        el.ownerDocument = this;
        return el;
      },
      createTextNode: function () { return {}; },
      createComment: function () { return {}; },
      createEvent: function () { return { initEvent: function () {} }; },
      dispatchEvent: function () { return true; },
      addEventListener: function () {},
      removeEventListener: function () {},
      querySelectorAll: function (selector) { return docEl.querySelectorAll(selector); },
      querySelector: function (selector) {
        return selector === "#jsa" && this.__jsa__ ? this.__jsa__ : docEl.querySelector(selector);
      },
      getElementById: function (id) {
        return id === "jsa" && this.__jsa__ ? this.__jsa__ : null;
      },
    };
    docEl.ownerDocument = doc;
    head.ownerDocument = doc;
    body.ownerDocument = doc;
    return doc;
  }

  // ===== Main document =====
  var doc__ = makeDocument__();

  // ===== Sandbox iframe =====
  var contentDoc__ = makeDocument__();
  // CSP meta
  var cspMeta__ = contentDoc__.createElement("meta");
  cspMeta__.setAttribute("http-equiv", "Content-Security-Policy");
  cspMeta__.setAttribute("content", "default-src 'none'; script-src 'unsafe-inline';");
  contentDoc__.head.appendChild(cspMeta__);

  // iframe element
  var jsaFrame__ = doc__.createElement("iframe");
  jsaFrame__.setAttribute("id", "jsa");
  jsaFrame__.setAttribute("sandbox", "allow-scripts allow-same-origin");
  jsaFrame__.style.cssText = "position: absolute; left: -9999px; top: -9999px;";
  jsaFrame__.srcdoc = "<!DOCTYPE html>\n<html>\n<head>\n"
    + "<meta http-equiv=\"Content-Security-Policy\""
    + " content=\"default-src 'none'; script-src 'unsafe-inline';\">\n"
    + "</head>\n<body></body>\n</html>";

  // Iframe content window — the challenge code runs here
  var contentWin__ = {
    Array: Array, Promise: Promise, Proxy: Proxy, Symbol: Symbol,
    Object: Object, JSON: JSON, Math: Math, Date: Date,
    String: String, Number: Number, Boolean: Boolean, RegExp: RegExp,
    Map: Map, Set: Set, WeakMap: WeakMap, WeakSet: WeakSet,
    Error: Error, TypeError: TypeError, RangeError: RangeError,
    ReferenceError: ReferenceError, SyntaxError: SyntaxError,
    EvalError: EvalError, URIError: URIError,
    Uint8Array: Uint8Array, Uint16Array: Uint16Array, Uint32Array: Uint32Array,
    Int8Array: Int8Array, Int16Array: Int16Array, Int32Array: Int32Array,
    Float32Array: Float32Array, Float64Array: Float64Array,
    ArrayBuffer: ArrayBuffer, DataView: DataView,
    TextEncoder: TextEncoder__,
    navigator: navigator__,
    document: contentDoc__,
    location: {
      href: "about:srcdoc", origin: "null", protocol: "about:",
      host: "", hostname: "", port: "", pathname: "srcdoc",
      search: "", hash: "",
    },
    btoa: function (v) { return __goBtoa(v); },
    atob: function (v) { return __goAtob(v); },
    setTimeout: function (fn) { if (typeof fn === "function") fn(); return 0; },
    clearTimeout: function () {},
    setInterval: function () { return 0; },
    clearInterval: function () {},
    addEventListener: function () {},
    removeEventListener: function () {},
    postMessage: function () {},
    getComputedStyle: function (el) {
      return el && el.style ? el.style
        : { getPropertyValue: function () { return ""; }, cssText: "" };
    },
    screen: {
      width: 1920, height: 1080, availWidth: 1920,
      availHeight: 1040, colorDepth: 24, pixelDepth: 24,
    },
    crypto: { subtle: { digest: function () { return Promise.resolve(new ArrayBuffer(32)); } } },
    performance: { now: function () { var t = Date.now(); return t % 1000 + Math.random(); } },
    console: {
      log: function () {}, warn: function () {}, error: function () {},
      info: function () {}, debug: function () {},
    },
    __jsaCallbacks__: {},
    // === Window identity checks ===
    constructor: function Window() {},
    navigator: navigator__,
  };
  contentWin__.self = contentWin__;
  contentWin__.window = contentWin__;
  contentWin__.top = globalThis;
  contentWin__.parent = globalThis;
  contentWin__[Symbol.toStringTag] = "Window";
  // Object.getOwnPropertyNames support for window property enumeration
  contentWin__.Window = function Window() {};
  contentWin__.Window.prototype = contentWin__;
  contentDoc__.defaultView = contentWin__;
  jsaFrame__.contentDocument = contentDoc__;
  jsaFrame__.contentWindow = contentWin__;

  doc__.body.appendChild(jsaFrame__);
  doc__.__jsa__ = jsaFrame__;

  // ===== Global property install =====
  function defProp__(obj, name, value) {
    Object.defineProperty(obj, name, { value: value, writable: true, configurable: true });
  }
  defProp__(globalThis, "window", globalThis);
  defProp__(globalThis, "self", globalThis);
  defProp__(globalThis, "top", globalThis);
  defProp__(globalThis, "document", doc__);
  defProp__(globalThis, "location", docLocation__);
  defProp__(globalThis, "navigator", navigator__);
  defProp__(globalThis, "TextEncoder", TextEncoder__);
  defProp__(globalThis, "Element", Element__);
  defProp__(globalThis, "HTMLElement", HTMLElement__);
  defProp__(globalThis, "HTMLDivElement", HTMLDivElement__);
  defProp__(globalThis, "HTMLIFrameElement", HTMLIFrameElement__);
  defProp__(globalThis, "HTMLScriptElement", HTMLScriptElement__);
  defProp__(globalThis, "NodeList", NodeList__);
  defProp__(globalThis, "__DDG_BE_VERSION__", "dev");
  defProp__(globalThis, "__DDG_FE_CHAT_HASH__", "hash");
  defProp__(globalThis, "Window", function Window() {});
  try { defProp__(globalThis.Window, "prototype", globalThis); } catch (e) {
    // Goja 的 Window 构造函数的 prototype 不可重新定义, 跳过
  }
  // Symbol.toStringTag: 让 Object.prototype.toString.call(window) === "[object Window]"
  if (typeof Symbol !== "undefined" && Symbol.toStringTag) {
    defProp__(globalThis, Symbol.toStringTag, "Window");
    defProp__(contentWin__, Symbol.toStringTag, "Window");
  }
  defProp__(globalThis, "btoa", function (v) { return __goBtoa(v); });
  defProp__(globalThis, "atob", function (v) { return __goAtob(v); });
  defProp__(globalThis, "getComputedStyle", function (el) {
    return el && el.style ? el.style
      : { getPropertyValue: function () { return ""; }, cssText: "" };
  });
  defProp__(globalThis, "setTimeout", function (fn) {
    if (typeof fn === "function") fn();
    return 0;
  });
  defProp__(globalThis, "clearTimeout", function () {});
  defProp__(globalThis, "setInterval", function () { return 0; });
  defProp__(globalThis, "clearInterval", function () {});
  defProp__(globalThis, "performance", {
    now: function () { var t = Date.now(); return t % 1000 + Math.random(); },
    timing: { navigationStart: Date.now() - 1000 },
    memory: { jsHeapSizeLimit: 2172649472, totalJSHeapSize: 10000000, usedJSHeapSize: 8000000 },
    timeOrigin: Date.now() - 1000,
  });
  defProp__(globalThis, "crypto", {
    subtle: { digest: function () { return Promise.resolve(new ArrayBuffer(32)); } },
    getRandomValues: function (arr) {
      for (var i = 0; i < arr.length; i++) arr[i] = Math.floor(Math.random() * 256);
      return arr;
    },
  });
  defProp__(globalThis, "screen", {
    width: 1920, height: 1080, availWidth: 1920,
    availHeight: 1040, colorDepth: 24, pixelDepth: 24,
  });
  defProp__(globalThis, "history", { length: 1, state: null, scrollRestoration: "auto" });
  defProp__(globalThis, "localStorage", (function () {
    var s = {};
    return {
      getItem: function (k) { return s[k] !== undefined ? s[k] : null; },
      setItem: function (k, v) { s[String(k)] = String(v); },
      removeItem: function (k) { delete s[String(k)]; },
      clear: function () { s = {}; },
      get length() { return Object.keys(s).length; },
      key: function (i) { return Object.keys(s)[i] || null; },
    };
  })());
  defProp__(globalThis, "sessionStorage", (function () {
    var s = {};
    return {
      getItem: function (k) { return s[k] !== undefined ? s[k] : null; },
      setItem: function (k, v) { s[String(k)] = String(v); },
      removeItem: function (k) { delete s[String(k)]; },
      clear: function () { s = {}; },
      get length() { return Object.keys(s).length; },
      key: function (i) { return Object.keys(s)[i] || null; },
    };
  })());
  defProp__(globalThis, "console", {
    log: function () {}, warn: function () {}, error: function () {},
    info: function () {}, debug: function () {},
  });
  defProp__(globalThis, "XMLHttpRequest", function () {
    this.open = function () {}; this.send = function () {}; this.setRequestHeader = function () {};
    this.abort = function () {}; this.readyState = 4; this.status = 200; this.responseText = "";
  });
  defProp__(globalThis, "fetch", function () {
    return Promise.resolve({
      ok: true,
      status: 200,
      json: function () { return Promise.resolve({}); },
      headers: { get: function () { return null; } },
    });
  });
  defProp__(globalThis, "URL", function (url) {
    var u = {
      href: url, protocol: "https:", host: "", hostname: "",
      port: "", pathname: "/", search: "", hash: "", origin: __goOrigin,
    };
    return u;
  });
  defProp__(globalThis, "URLSearchParams", function () {
    this.get = function () { return null; };
    this.set = function () {};
    this.keys = function () { return []; };
  });
  defProp__(globalThis, "requestAnimationFrame", function (fn) {
    if (typeof fn === "function") fn(0);
    return 0;
  });
  defProp__(globalThis, "cancelAnimationFrame", function () {});
  defProp__(globalThis, "matchMedia", function () {
    return {
      matches: false,
      addListener: function () {},
      removeListener: function () {},
      addEventListener: function () {},
      removeEventListener: function () {},
    };
  });
  defProp__(globalThis, "ResizeObserver", function () {
    this.observe = function () {};
    this.disconnect = function () {};
    this.unobserve = function () {};
  });
  defProp__(globalThis, "IntersectionObserver", function () {
    this.observe = function () {};
    this.disconnect = function () {};
    this.unobserve = function () {};
  });
  defProp__(globalThis, "MutationObserver", function () {
    this.observe = function () {};
    this.disconnect = function () {};
    this.takeRecords = function () { return []; };
  });
  defProp__(globalThis, "Image", function () {
    var img = {
      width: 0, height: 0, src: "", onload: null, onerror: null,
      naturalWidth: 0, naturalHeight: 0, complete: false,
    };
    return img;
  });

  // Fix constructor .name for challenge compatibility
  try {
    Object.defineProperty(NodeList, "name", { value: "NodeList", configurable: true });
  } catch (e) {}
  try {
    Object.defineProperty(Element, "name", { value: "Element", configurable: true });
  } catch (e) {}
  try {
    Object.defineProperty(HTMLElement, "name", { value: "HTMLElement", configurable: true });
  } catch (e) {}
  try {
    Object.defineProperty(HTMLDivElement, "name", { value: "HTMLDivElement", configurable: true });
  } catch (e) {}
  try {
    Object.defineProperty(HTMLIFrameElement, "name", {
      value: "HTMLIFrameElement",
      configurable: true,
    });
  } catch (e) {}
  try {
    Object.defineProperty(HTMLScriptElement, "name", {
      value: "HTMLScriptElement",
      configurable: true,
    });
  } catch (e) {}
})();
"""
)
