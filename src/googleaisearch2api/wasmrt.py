"""QuickJS-in-WASI bridge: evaluate JavaScript inside wasmtime from Python.

Artifacts (``src/googleaisearch2api/vendor/``): ``quickjs.wasm`` — QuickJS-NG
compiled with wasi-sdk to wasm32-wasip1 in reactor mode — plus the ``crypto``
and ``encoding`` native extensions (WebCrypto / TextEncoder). All three come
from the npm package ``quickjs-wasi`` (vercel-labs/quickjs-wasi), pinned and
reproducible via ``scripts/build_wasm.sh``.

Main entrypoint::

    eval_js(code) -> str   # captured stdout/console output of the snippet

The runtime is a lazy singleton: a wasmtime ``Store`` is strictly
single-threaded, so all access is serialized under one re-entrant lock.
Only ``wasmtime`` is imported beyond the stdlib.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time

import wasmtime as wt

from . import wasmrt_boot
from .wasmrt_ext import load_extension

VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")

# sha256 pins of the vendored artifacts (see scripts/build_wasm.sh).
_ARTIFACT_SHA256 = {
    "quickjs.wasm": "5d2c236622a4c0f64e27657f246f1ee0ea6e4ba637462137da4fdeacdc53d20c",
    "crypto.so": "fa26b6fafda251c3b1442dcc4c876b3ac73689639519f4ddfc2c410ebdb5b7c2",
    "encoding.so": "d501aa2a8410ee07c196103f0d169b226ec61c922883713510fd723fe14036c3",
}
_EXTENSION_FILES = {"crypto": "crypto.so", "encoding": "encoding.so"}

INTRINSIC_ALL = 0xFFFFFFFF
_JS_EVAL_GLOBAL = 0
_DEFAULT_TIMEOUT_S = 30.0
_MAX_JOB_ITERATIONS = 100_000

I32, I64 = wt.ValType.i32(), wt.ValType.i64()


class _QuickJsRuntime:
    """One QuickJS context inside one wasmtime instance. Guard with the lock."""

    __slots__ = ("engine", "store", "linker", "instance", "exports", "mem", "out", "deadline")

    def __init__(self) -> None:
        self.engine = wt.Engine()
        self.store = wt.Store(self.engine)
        self.linker = wt.Linker(self.engine)
        self.out = bytearray()
        self.deadline: float | None = None
        self._define_imports()
        self.instance = self.linker.instantiate(self.store, self._load_module())
        self.exports = self.instance.exports(self.store)
        self.mem = self.exports["memory"]
        self.exports["_initialize"](self.store)
        rc = self.exports["qjs_init2"](self.store, INTRINSIC_ALL)
        if rc != 0:
            raise RuntimeError(f"quickjs context init failed (rc={rc})")
        self.exports["qjs_set_interrupt_handler"](self.store, 1)
        self._register_print()
        for name, filename in _EXTENSION_FILES.items():
            load_extension(self, os.path.join(VENDOR_DIR, filename), name)
        if not self._eval_ok(wasmrt_boot.PRELUDE_SOURCE):
            raise RuntimeError("quickjs prelude failed to evaluate")
        if not self._eval_ok(wasmrt_boot.RESULT_CAPTURE_SOURCE):
            raise RuntimeError("result-capture hook failed to install")

    # ---- artifact handling --------------------------------------------------
    def _load_module(self) -> wt.Module:
        path = os.path.join(VENDOR_DIR, "quickjs.wasm")
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest != _ARTIFACT_SHA256["quickjs.wasm"]:
            raise RuntimeError(f"vendored quickjs.wasm checksum mismatch: {digest}")
        return wt.Module.from_file(self.engine, path)

    # ---- host imports --------------------------------------------------------
    def _define_imports(self) -> None:
        def d(mod: str, name: str, cb, params, results, acc: bool = True) -> None:
            self.linker.define_func(mod, name, wt.FuncType(params, results), cb, access_caller=acc)

        d("wasi_snapshot_preview1", "clock_time_get", self._w_clock, [I32, I64, I32], [I32])
        d("wasi_snapshot_preview1", "fd_close", lambda c, fd: 52, [I32], [I32])
        d("wasi_snapshot_preview1", "fd_fdstat_get", self._w_fdstat, [I32, I32], [I32])
        d(
            "wasi_snapshot_preview1",
            "fd_seek",
            lambda c, a, b, cc, dd: 52,
            [I32, I64, I32, I32],
            [I32],
        )
        d("wasi_snapshot_preview1", "fd_write", self._w_write, [I32, I32, I32, I32], [I32])
        d("wasi_snapshot_preview1", "random_get", self._w_random, [I32, I32], [I32])
        d("env", "host_get_timezone_offset", lambda c, hi, lo: 0, [I32, I32], [I32])
        d("env", "host_interrupt", self._w_interrupt, [], [I32])
        d("env", "host_promise_rejection", self._w_rejection, [I32, I32, I32], [])
        d("env", "host_module_normalize", lambda c, a, b: 0, [I32, I32], [I32], acc=False)
        d("env", "host_module_load", lambda c, a, b: 0, [I32, I32], [I32], acc=False)
        d("env", "host_call", self._w_call, [I32, I32, I32, I32, I32], [I32])

    def _w_clock(self, caller, clock_id, precision, out_ptr):
        caller.get("memory").write(caller, struct.pack("<Q", int(time.time() * 1e9)), out_ptr)
        return 0

    def _w_fdstat(self, caller, fd, stat_ptr):
        if fd in (1, 2):
            mem = caller.get("memory")
            mem.write(caller, bytes(24), stat_ptr)
            mem.write(caller, bytes([2]), stat_ptr)
            return 0
        return 8

    def _w_write(self, caller, fd, iovs, n, nwritten):
        if fd not in (1, 2):
            return 8
        mem = caller.get("memory")
        total = 0
        for i in range(n):
            ptr, length = struct.unpack("<II", mem.read(caller, iovs + i * 8, iovs + i * 8 + 8))
            self.out.extend(mem.read(caller, ptr, ptr + length))
            total += length
        mem.write(caller, struct.pack("<I", total), nwritten)
        return 0

    def _w_random(self, caller, ptr, n):
        caller.get("memory").write(caller, os.urandom(n), ptr)
        return 0

    def _w_interrupt(self, caller):
        if self.deadline is not None and time.monotonic() > self.deadline:
            return 1
        return 0

    def _w_rejection(self, caller, promise_ptr, reason_ptr, is_handled):
        self.out.extend(b'\n{"__unhandled_rejection": ')
        self.out.extend(json.dumps(self._value_to_str(reason_ptr)).encode())
        self.out.extend(b"}\n")
        self.exports["qjs_free_value"](self.store, promise_ptr)
        self.exports["qjs_free_value"](self.store, reason_ptr)

    def _w_call(self, caller, name_ptr, name_len, this_ptr, argc, argv_ptr):
        mem = caller.get("memory")
        name = bytes(mem.read(caller, name_ptr, name_ptr + name_len)).decode("utf-8", "replace")
        if name == "wasmrt_print":
            for i in range(argc):
                p = struct.unpack("<I", mem.read(caller, argv_ptr + i * 4, argv_ptr + i * 4 + 4))[0]
                self.out.extend(self._value_to_str(p).encode("utf-8", "replace"))
        else:
            self._throw_js(f"unknown host function: {name}")
            return 0
        return self.exports["qjs_get_undefined"](caller)

    # ---- memory internals -----------------------------------------------------
    def _write_cstr(self, data: bytes) -> int:
        p = self.exports["wasm_malloc"](self.store, len(data) + 1)
        self.mem.write(self.store, data + b"\0", p)
        return p

    def _value_to_str(self, vptr: int) -> str:
        sptr = self.exports["qjs_get_string"](self.store, vptr)
        if not sptr:
            return ""
        raw = bytes(self.mem.read(self.store, sptr, sptr + (1 << 20)))
        text = raw.split(b"\0")[0].decode("utf-8", "replace")
        self.exports["qjs_free_cstring"](self.store, sptr)
        return text

    def _throw_js(self, message: str) -> None:
        data = message.encode("utf-8")
        mptr = self._write_cstr(data)
        mval = self.exports["qjs_new_string"](self.store, mptr, len(data))
        self.exports["wasm_free"](self.store, mptr)
        err = self.exports["qjs_new_error"](self.store)
        kptr = self._write_cstr(b"message")
        self.exports["qjs_set_prop_string"](self.store, err, kptr, mval)
        self.exports["wasm_free"](self.store, kptr)
        self.exports["qjs_free_value"](self.store, mval)
        self.exports["qjs_throw"](self.store, err)
        self.exports["qjs_free_value"](self.store, err)

    def _register_print(self) -> None:
        name = "wasmrt_print"
        nptr = self._write_cstr(name.encode())
        fn = self.exports["qjs_new_host_function"](self.store, nptr, len(name), 0)
        self.exports["wasm_free"](self.store, nptr)
        glob = self.exports["qjs_get_global"](self.store)
        gptr = self._write_cstr(name.encode())
        self.exports["qjs_set_prop_string"](self.store, glob, gptr, fn)
        self.exports["wasm_free"](self.store, gptr)
        self.exports["qjs_free_value"](self.store, fn)
        self.exports["qjs_free_value"](self.store, glob)

    # ---- evaluation -----------------------------------------------------------
    def _eval_raw(self, code: str) -> int:
        data = code.encode("utf-8")
        p = self._write_cstr(data)
        v = self.exports["qjs_eval"](self.store, p, len(data), 0, _JS_EVAL_GLOBAL) & 0xFFFFFFFF
        self.exports["wasm_free"](self.store, p)
        return v

    def _eval_ok(self, code: str) -> bool:
        v = self._eval_raw(code)
        ok = not self.exports["qjs_is_exception"](self.store, v)
        self.exports["qjs_free_value"](self.store, v)
        return ok

    def _drain_jobs(self) -> None:
        for _ in range(_MAX_JOB_ITERATIONS):
            if not self.exports["qjs_is_job_pending"](self.store):
                return
            self.exports["qjs_execute_pending_job"](self.store)

    def eval_js(self, code: str) -> str:
        self.out.clear()
        wrapped = (
            "try { var __wv = (0, eval)("
            + wasmrt_boot.js_string_literal(code)
            + "); __wasmrt_out(__wv); } "
            "catch (e) { __wasmrt_out({ __error: "
            "(e && e.message) ? ('Error: ' + e.message) : String(e) }); }"
        )
        v = self._eval_raw(wrapped)
        if self.exports["qjs_is_exception"](self.store, v):
            exc = self.exports["qjs_get_exception"](self.store)
            text = self._value_to_str(exc)
            self.exports["qjs_free_value"](self.store, exc)
            self.exports["qjs_free_value"](self.store, v)
            raise ValueError(f"JavaScript evaluation failed: {text}")
        self.exports["qjs_free_value"](self.store, v)
        self._drain_jobs()
        out_text = bytes(self.out).decode("utf-8", "replace").rstrip()
        for line in out_text.splitlines():
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and "__error" in payload:
                raise ValueError(f"JavaScript evaluation failed: {payload['__error']}")
        return out_text


_runtime: _QuickJsRuntime | None = None
_lock = threading.RLock()


def _get_runtime() -> _QuickJsRuntime:
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = _QuickJsRuntime()
        return _runtime


def eval_js(code: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> str:
    """Evaluate ``code`` inside QuickJS and return captured stdout/console output.

    The completion value is printed at the end (strings raw, other values as
    JSON); asynchronous results settle before returning. Raises ``ValueError``
    when the JS throws — synchronously or from an async continuation — or when
    ``timeout_s`` expires mid-execution. ``timeout_s <= 0`` disables the guard.
    """
    with _lock:
        rt = _get_runtime()
        previous = rt.deadline
        rt.deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
        try:
            return rt.eval_js(code)
        finally:
            rt.deadline = previous
