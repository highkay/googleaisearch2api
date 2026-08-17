"""Native QuickJS extension (.so) loading via the wasm dylink.0 protocol.

The ``crypto`` and ``encoding`` extensions ship as wasm side modules built with
wasi-sdk (`-Wl,--shared` style). They import memory, the indirect function
table, the stack pointer and symbol functions from the main ``quickjs.wasm``
instance, and export an init function called with the live context/runtime
pointers.
"""

from __future__ import annotations

import os

import wasmtime as wt

I32 = wt.ValType.i32()


def _uldr(data: bytes, i: int) -> tuple[int, int]:
    v, shift = 0, 0
    while True:
        b = data[i]
        i += 1
        v |= (b & 0x7F) << shift
        if not (b & 0x80):
            return v, i
        shift += 7


def _parse_dylink(data: bytes) -> dict[str, int] | None:
    """Return {memory_size, table_size} from the dylink.0 section of a .so."""
    if data[:4] != b"\0asm":
        return None
    i = 8
    while i < len(data):
        sec_id = data[i]
        i += 1
        payload_len, i = _uldr(data, i)
        if sec_id == 0:  # custom section
            name_len, j = _uldr(data, i)
            name = data[j : j + name_len].decode("utf-8", "replace")
            content = data[j + name_len : i + payload_len]
            if name == "dylink.0":
                out: dict[str, int] = {"memory_size": 0, "table_size": 0}
                o = 0
                while o < len(content):
                    st = content[o]
                    o += 1
                    ssz, o = _uldr(content, o)
                    sub = content[o : o + ssz]
                    o += ssz
                    if st == 1:  # WASM_DYLINK_MEM_INFO
                        k = 0
                        out["memory_size"], k = _uldr(sub, k)
                        _, k = _uldr(sub, k)  # memory alignment
                        out["table_size"], k = _uldr(sub, k)
                return out
        i += payload_len
    return None


def load_extension(rt, path: str, name: str) -> None:
    """Instantiate ``path`` into ``rt``'s wasmtime store and call its init.

    ``rt`` is the module-private ``_QuickJsRuntime``; accessing its internals
    here keeps the loader self-contained while staying inside the package.
    """
    info = _parse_dylink(open(path, "rb").read())
    if not info:
        raise RuntimeError(f"extension {name}: no dylink.0 section")
    module = wt.Module.from_file(rt.engine, path)
    ex = rt.exports
    mem = ex["memory"]
    table = ex["__indirect_function_table"]

    mem_base = 0
    if info["memory_size"] > 0:
        mem_base = ex["malloc"](rt.store, info["memory_size"])
        if not mem_base:
            raise RuntimeError(f"extension {name}: static-data allocation failed")
        mem.write(rt.store, bytes(info["memory_size"]), mem_base)
    table_base = table.size(rt.store)
    if info["table_size"] > 0:
        table.grow(rt.store, info["table_size"], None)

    linker = wt.Linker(rt.engine)
    linker.define(rt.store, "env", "memory", mem)
    linker.define(rt.store, "env", "__indirect_function_table", table)
    linker.define(rt.store, "env", "__stack_pointer", ex["__stack_pointer"])
    linker.define(
        rt.store, "env", "__memory_base", wt.Global(rt.store, wt.GlobalType(I32, False), mem_base)
    )
    linker.define(
        rt.store, "env", "__table_base", wt.Global(rt.store, wt.GlobalType(I32, False), table_base)
    )

    reserved = {
        "memory",
        "__indirect_function_table",
        "__stack_pointer",
        "__memory_base",
        "__table_base",
    }
    main_names = set(ex.keys())
    for imp in module.imports:
        ft = imp.type
        if imp.module == "env" and imp.name not in reserved:
            target = ex[imp.name] if imp.name in main_names else None
            if isinstance(target, wt.Func):

                def mk(t, has_ret):
                    def cb(caller, *args):
                        return t(caller, *args) if has_ret else t(caller, *args) or None

                    return cb

                linker.define_func(
                    "env",
                    imp.name,
                    wt.FuncType(ft.params, ft.results),
                    mk(target, bool(ft.results)),
                    access_caller=True,
                )
            else:

                def mk_stub(nm):
                    def cb(*a):
                        raise wt.Trap(f"extension unresolved import: {nm}")

                    return cb

                linker.define_func(
                    "env",
                    imp.name,
                    wt.FuncType(ft.params, ft.results),
                    mk_stub(imp.name),
                    access_caller=False,
                )
        elif imp.module == "wasi_snapshot_preview1" and imp.name == "random_get":

            def rnd(caller, ptr, n):
                mem.write(caller, os.urandom(n), ptr)
                return 0

            linker.define_func(
                "wasi_snapshot_preview1",
                "random_get",
                wt.FuncType(ft.params, ft.results),
                rnd,
                access_caller=True,
            )
        elif imp.module == "wasi_snapshot_preview1":

            def nosys(*a, _imp=imp):
                raise wt.Trap(f"extension wasi call unsupported: {_imp.name}")

            linker.define_func(
                "wasi_snapshot_preview1",
                imp.name,
                wt.FuncType(ft.params, ft.results),
                nosys,
                access_caller=False,
            )
        elif imp.module.startswith("GOT.func"):
            target = ex[imp.name] if imp.name in main_names else None
            if isinstance(target, wt.Func):
                idx = table.size(rt.store)
                table.grow(rt.store, 1, None)
                table.set(rt.store, idx, target)
                val = idx
            else:
                val = 0
            linker.define(
                rt.store, imp.module, imp.name, wt.Global(rt.store, wt.GlobalType(I32, True), val)
            )
        elif imp.module.startswith("GOT.mem"):
            linker.define(
                rt.store, imp.module, imp.name, wt.Global(rt.store, wt.GlobalType(I32, True), 0)
            )
        elif imp.module == "env":
            continue  # reserved items already defined
        else:
            raise RuntimeError(f"extension {name}: unhandled import namespace {imp.module}")

    einst = linker.instantiate(rt.store, module)
    eex = einst.exports(rt.store)
    if "__wasm_apply_data_relocs" in eex:
        eex["__wasm_apply_data_relocs"](rt.store)
    if "__wasm_call_ctors" in eex:
        eex["__wasm_call_ctors"](rt.store)
    init_fn = eex[f"qjs_ext_{name}_init"]
    rc = init_fn(rt.store, ex["qjs_get_context_ptr"](rt.store), ex["qjs_get_runtime_ptr"](rt.store))
    if rc not in (0, None):
        raise RuntimeError(f"extension {name} init failed (rc={rc})")
