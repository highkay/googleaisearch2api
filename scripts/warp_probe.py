"""Tunneled SOCKS5 probe engine for the warp-plus health monitor.

Completes a REAL HTTPS request through the container's SOCKS5 port with remote
DNS (socks5h semantics) against the Cloudflare connectivity trace endpoint, so
a container whose port still accepts TCP but whose tunnel is dead is caught.

Transports: curl_cffi when importable, else a stdlib-only SOCKS5 handshake +
CONNECT + TLS fallback.
"""

from __future__ import annotations

import socket
import ssl
import time
from dataclasses import dataclass

TRACE_URL = "https://connectivity.cloudflareclient.com/cdn-cgi/trace"
TRACE_HOST = "connectivity.cloudflareclient.com"
TRACE_PATH = "/cdn-cgi/trace"

SOCKS5_GREETING = b"\x05\x01\x00"


@dataclass(slots=True)
class ProbeResult:
    ok: bool
    client: str | None = None
    status_code: int | None = None
    elapsed_s: float | None = None
    error: str | None = None
    evidence: str | None = None


def probe_response_ok(status_code: int | None, body: str) -> bool:
    if status_code == 200:
        return True
    return "warp=" in (body or "")


def socks5_connect_request(host: str, port: int) -> bytes:
    """SOCKS5 CONNECT with ATYP=domain so the proxy does the (remote) DNS."""
    encoded = host.encode("ascii")
    if len(encoded) > 255:
        raise ValueError("SOCKS5 domain exceeds 255 bytes")
    if not 0 < port <= 65535:
        raise ValueError("SOCKS5 port out of range")
    return b"\x05\x01\x00\x03" + bytes([len(encoded)]) + encoded + port.to_bytes(2, "big")


def _recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        piece = connection.recv(length - len(chunks))
        if not piece:
            raise OSError("connection closed mid-read")
        chunks.extend(piece)
    return bytes(chunks)


def _split_http_response(raw: bytes) -> tuple[int | None, str]:
    header, _blank, body = raw.partition(b"\r\n\r\n")
    parts = header.split(b"\r\n", 1)[0].split()
    status: int | None = None
    if len(parts) >= 2 and parts[1].isdigit():
        status = int(parts[1])
    return status, body.decode("utf-8", "replace")


def _probe_via_curl_cffi(host_port: int, *, timeout: float) -> ProbeResult:
    from curl_cffi import requests as cffi_requests
    from curl_cffi.requests.exceptions import RequestException as CffiRequestError

    started = time.monotonic()
    try:
        response = cffi_requests.get(
            TRACE_URL,
            proxy=f"socks5h://127.0.0.1:{host_port}",
            impersonate="chrome",
            timeout=timeout,
        )
        text = response.text or ""
        status = response.status_code
    except CffiRequestError as exc:
        return ProbeResult(
            ok=False,
            client="curl_cffi",
            elapsed_s=round(time.monotonic() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
    elapsed = round(time.monotonic() - started, 3)
    ok = probe_response_ok(status, text)
    return ProbeResult(
        ok=ok,
        client="curl_cffi",
        status_code=status,
        elapsed_s=elapsed,
        error=None if ok else f"trace status={status} without warp marker",
        evidence=text[:200],
    )


def _stdlib_failure(started: float, message: str) -> ProbeResult:
    return ProbeResult(
        ok=False,
        client="stdlib",
        elapsed_s=round(time.monotonic() - started, 3),
        error=message,
    )


def _probe_via_stdlib_socks(host_port: int, *, timeout: float) -> ProbeResult:
    started = time.monotonic()
    try:
        with socket.create_connection(("127.0.0.1", host_port), timeout=timeout) as raw:
            raw.settimeout(timeout)
            raw.sendall(SOCKS5_GREETING)
            greeting = _recv_exact(raw, 2)
            if greeting[0] != 0x05 or greeting[1] != 0x00:
                return _stdlib_failure(started, f"socks5 handshake rejected: {greeting.hex()}")
            raw.sendall(socks5_connect_request(TRACE_HOST, 443))
            header = _recv_exact(raw, 4)
            if header[0] != 0x05 or header[1] != 0x00:
                return _stdlib_failure(started, f"socks5 CONNECT rejected: {header.hex()}")
            atyp = header[3]
            if atyp == 0x01:
                _recv_exact(raw, 4)
            elif atyp == 0x03:
                _recv_exact(raw, _recv_exact(raw, 1)[0])
            elif atyp == 0x04:
                _recv_exact(raw, 16)
            else:
                return _stdlib_failure(started, f"socks5 unexpected atyp 0x{atyp:02x}")
            _recv_exact(raw, 2)
            context = ssl.create_default_context()
            with context.wrap_socket(raw, server_hostname=TRACE_HOST) as tls:
                request = (
                    f"GET {TRACE_PATH} HTTP/1.1\r\nHost: {TRACE_HOST}\r\n"
                    "User-Agent: warp-health/1.0\r\nAccept: */*\r\nConnection: close\r\n\r\n"
                )
                tls.sendall(request.encode("ascii"))
                payload = bytearray()
                while True:
                    piece = tls.recv(8192)
                    if not piece:
                        break
                    payload.extend(piece)
                    if len(payload) > 65536:
                        break
    except Exception as exc:
        # Boundary catch: any transport/TLS failure IS the probe evidence.
        return _stdlib_failure(started, f"{type(exc).__name__}: {exc}")
    status, text = _split_http_response(bytes(payload))
    elapsed = round(time.monotonic() - started, 3)
    ok = probe_response_ok(status, text)
    return ProbeResult(
        ok=ok,
        client="stdlib",
        status_code=status,
        elapsed_s=elapsed,
        error=None if ok else f"trace status={status} without warp marker",
        evidence=text[:200],
    )


def probe_warp_exit(host_port: int, *, timeout: float = 8.0) -> ProbeResult:
    try:
        return _probe_via_curl_cffi(host_port, timeout=timeout)
    except (ImportError, ValueError):
        return _probe_via_stdlib_socks(host_port, timeout=timeout)
