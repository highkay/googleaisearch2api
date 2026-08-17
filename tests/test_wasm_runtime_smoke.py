"""Smoke tests for the QuickJS WASI runtime bridge (wasmrt).

Given: the vendored quickjs.wasm (+ crypto/encoding extensions) and the
wasmtime dependency installed.
When: JS code is evaluated through googleaisearch2api.wasmrt.eval_js.
Then: stdout/console output is captured and the QuickJS context supports the
features the Duck.ai challenge solver needs.
"""

import pytest

from googleaisearch2api import wasmrt

SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

JS_SHA256_POLYFILL = r"""
function js_sha256_hex(bytes) {
  function rotr(x, n) { return (x >>> n) | (x << (32 - n)); }
  var K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  var H = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ];
  var m = new Uint8Array(((bytes.length + 9 + 63) >> 6) << 6);
  m.set(bytes);
  var bitLen = bytes.length * 8;
  m[bytes.length] = 0x80;
  for (var bi = 0; bi < 4; bi++) m[m.length - 1 - bi] = (bitLen >>> (8 * bi)) & 0xff;
  for (var off = 0; off < m.length; off += 64) {
    var w = new Array(64);
    for (var i = 0; i < 16; i++) {
      w[i] = (m[off + 4 * i] << 24) | (m[off + 4 * i + 1] << 16)
           | (m[off + 4 * i + 2] << 8) | m[off + 4 * i + 3];
    }
    for (i = 16; i < 64; i++) {
      var s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      var s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    var a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (i = 0; i < 64; i++) {
      var S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      var ch = (e & f) ^ (~e & g);
      var t1 = (h + S1 + ch + K[i] + w[i]) | 0;
      var S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      var maj = (a & b) ^ (a & c) ^ (b & c);
      var t2 = (S0 + maj) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    H[0] = (H[0] + a) | 0; H[1] = (H[1] + b) | 0; H[2] = (H[2] + c) | 0; H[3] = (H[3] + d) | 0;
    H[4] = (H[4] + e) | 0; H[5] = (H[5] + f) | 0; H[6] = (H[6] + g) | 0; H[7] = (H[7] + h) | 0;
  }
  return H.map(function (x) { return ("0000000" + (x >>> 0).toString(16)).slice(-8); }).join("");
}
js_sha256_hex(new TextEncoder().encode("abc"));
"""


def test_eval_arithmetic_result():
    assert wasmrt.eval_js("1+2") == "3"


def test_sha256_via_native_webcrypto_inside_wasm():
    code = r"""
(async function () {
  var digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode("abc"));
  return Array.from(new Uint8Array(digest))
    .map(function (b) { return b.toString(16).padStart(2, "0"); })
    .join("");
})();
"""
    assert wasmrt.eval_js(code) == SHA256_ABC


def test_sha256_via_js_polyfill_inside_wasm():
    assert wasmrt.eval_js(JS_SHA256_POLYFILL) == SHA256_ABC


def test_console_log_is_captured_to_stdout():
    assert wasmrt.eval_js("console.log('probe-42'); 7") == "probe-42\n7"


def test_async_promise_chain_completes():
    code = (
        "(async function () { "
        "var a = await Promise.resolve(20); var b = await Promise.resolve(22); return a + b; })()"
    )
    assert wasmrt.eval_js(code) == "42"


def test_btoa_atob_roundtrip():
    assert wasmrt.eval_js("atob(btoa('duck-challenge'))") == "duck-challenge"


def test_json_and_error_primitives():
    assert wasmrt.eval_js("JSON.parse('{\"x\": 41}').x + 1") == "42"
    assert wasmrt.eval_js("new Error('x').name") == "Error"


def test_intrinsic_coverage_table():
    code = (
        "[typeof Map, typeof Set, typeof WeakRef, typeof BigInt, typeof Promise, "
        "typeof Date, typeof JSON, typeof TextEncoder].join(',')"
    )
    assert (
        wasmrt.eval_js(code)
        == "function,function,function,function,function,function,object,function"
    )


def test_js_throw_raises_python_error():
    with pytest.raises(ValueError, match="kaputt-7"):
        wasmrt.eval_js("throw new Error('kaputt-7')")
