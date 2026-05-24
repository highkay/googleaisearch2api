from __future__ import annotations

import ipaddress
import select
import socket
import socketserver
import threading
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit, urlunsplit

SOCKS_PROXY_SCHEMES = {"socks5", "socks5h"}
DEFAULT_SOCKS5_PORT = 1080
HTTP_HEADER_LIMIT = 64 * 1024
RELAY_BUFFER_SIZE = 64 * 1024
SOCKET_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class SocksProxyTarget:
    server: str
    host: str
    port: int
    username: str | None = None
    password: str | None = None
    remote_dns: bool = True


def is_socks_proxy_server(proxy_server: str) -> bool:
    return urlsplit(proxy_server).scheme.lower() in SOCKS_PROXY_SCHEMES


def build_socks_proxy_target(
    proxy_server: str,
    *,
    username: str | None = None,
    password: str | None = None,
) -> SocksProxyTarget:
    parsed = urlsplit(proxy_server)
    scheme = parsed.scheme.lower()
    if scheme not in SOCKS_PROXY_SCHEMES:
        raise ValueError(f"Unsupported SOCKS proxy scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError("SOCKS proxy host is required.")

    port = parsed.port or DEFAULT_SOCKS5_PORT
    netloc = parsed.hostname
    if ":" in netloc and not netloc.startswith("["):
        netloc = f"[{netloc}]"
    netloc = f"{netloc}:{port}"

    return SocksProxyTarget(
        server=urlunsplit((scheme, netloc, "", "", "")),
        host=parsed.hostname,
        port=port,
        username=username or (unquote(parsed.username) if parsed.username else None),
        password=password or (unquote(parsed.password) if parsed.password else None),
        remote_dns=scheme == "socks5h",
    )


class LocalSocksProxyBridge:
    def __init__(self, target: SocksProxyTarget) -> None:
        self._target = target
        self._server: _ThreadingHttpProxyServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def server_url(self) -> str:
        if self._server is None:
            raise RuntimeError("SOCKS proxy bridge has not been started.")
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _ThreadingHttpProxyServer(
            ("127.0.0.1", 0),
            _HttpConnectProxyHandler,
            self._target,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5)


class _ThreadingHttpProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[socketserver.BaseRequestHandler],
        target: SocksProxyTarget,
    ) -> None:
        self.target = target
        super().__init__(server_address, handler_class)


class _HttpConnectProxyHandler(socketserver.BaseRequestHandler):
    server: _ThreadingHttpProxyServer

    def handle(self) -> None:
        upstream: socket.socket | None = None
        sent_success = False
        try:
            self.request.settimeout(SOCKET_TIMEOUT_SECONDS)
            header = _read_http_header(self.request)
            method, authority = _parse_http_request_line(header)
            if method != "CONNECT":
                _send_http_error(self.request, 405, "Method Not Allowed")
                return

            host, port = _parse_authority(authority)
            upstream = _open_socks_connection(self.server.target, host, port)
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            sent_success = True
            _relay(self.request, upstream)
        except Exception:
            if not sent_success:
                _send_http_error(self.request, 502, "Bad Gateway")
        finally:
            if upstream is not None:
                upstream.close()


def _read_http_header(sock: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Client closed connection before sending HTTP headers.")
        data += chunk
        if len(data) > HTTP_HEADER_LIMIT:
            raise ValueError("HTTP proxy request headers are too large.")
    return data.split(b"\r\n\r\n", 1)[0]


def _parse_http_request_line(header: bytes) -> tuple[str, str]:
    first_line = header.split(b"\r\n", 1)[0].decode("iso-8859-1")
    parts = first_line.split()
    if len(parts) != 3:
        raise ValueError("Invalid HTTP proxy request line.")
    return parts[0].upper(), parts[1]


def _parse_authority(authority: str) -> tuple[str, int]:
    parsed = urlsplit(f"//{authority}")
    if not parsed.hostname or parsed.port is None:
        raise ValueError("CONNECT authority must include host and port.")
    return parsed.hostname, parsed.port


def _send_http_error(sock: socket.socket, status_code: int, reason: str) -> None:
    body = reason.encode("ascii")
    try:
        sock.sendall(
            f"HTTP/1.1 {status_code} {reason}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n".encode("ascii")
            + body
        )
    except OSError:
        pass


def _open_socks_connection(
    target: SocksProxyTarget,
    destination_host: str,
    destination_port: int,
) -> socket.socket:
    sock = socket.create_connection(
        (target.host, target.port),
        timeout=SOCKET_TIMEOUT_SECONDS,
    )
    try:
        _negotiate_socks5(sock, target)
        _send_socks5_connect(sock, target, destination_host, destination_port)
    except Exception:
        sock.close()
        raise
    sock.settimeout(None)
    return sock


def _negotiate_socks5(sock: socket.socket, target: SocksProxyTarget) -> None:
    methods = [0x00]
    if target.username is not None or target.password is not None:
        methods.insert(0, 0x02)
    sock.sendall(bytes([0x05, len(methods), *methods]))

    version, selected_method = _recv_exact(sock, 2)
    if version != 0x05:
        raise OSError("SOCKS5 proxy returned an invalid greeting.")
    if selected_method == 0xFF:
        raise OSError("SOCKS5 proxy did not accept any authentication method.")
    if selected_method == 0x02:
        _authenticate_socks5(sock, target)
    elif selected_method != 0x00:
        raise OSError("SOCKS5 proxy selected an unsupported authentication method.")


def _authenticate_socks5(sock: socket.socket, target: SocksProxyTarget) -> None:
    username = (target.username or "").encode()
    password = (target.password or "").encode()
    if len(username) > 255 or len(password) > 255:
        raise ValueError("SOCKS5 username and password must be 255 bytes or shorter.")

    sock.sendall(
        b"\x01"
        + bytes([len(username)])
        + username
        + bytes([len(password)])
        + password
    )
    version, status = _recv_exact(sock, 2)
    if version != 0x01 or status != 0x00:
        raise OSError("SOCKS5 proxy authentication failed.")


def _send_socks5_connect(
    sock: socket.socket,
    target: SocksProxyTarget,
    destination_host: str,
    destination_port: int,
) -> None:
    sock.sendall(
        b"\x05\x01\x00"
        + _encode_socks5_address(destination_host, remote_dns=target.remote_dns)
        + destination_port.to_bytes(2, "big")
    )
    version, reply_code, _, address_type = _recv_exact(sock, 4)
    if version != 0x05:
        raise OSError("SOCKS5 proxy returned an invalid connect response.")
    _read_socks5_address(sock, address_type)
    _recv_exact(sock, 2)
    if reply_code != 0x00:
        raise OSError(f"SOCKS5 proxy connect failed with reply code {reply_code}.")


def _encode_socks5_address(host: str, *, remote_dns: bool) -> bytes:
    ip_address = _parse_ip_address(host)
    if ip_address is None and not remote_dns:
        resolved_host = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)[0][4][0]
        ip_address = ipaddress.ip_address(resolved_host)

    if ip_address is not None:
        if ip_address.version == 4:
            return b"\x01" + ip_address.packed
        return b"\x04" + ip_address.packed

    domain = host.encode("idna")
    if len(domain) > 255:
        raise ValueError("SOCKS5 destination hostname is too long.")
    return b"\x03" + bytes([len(domain)]) + domain


def _parse_ip_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _read_socks5_address(sock: socket.socket, address_type: int) -> bytes:
    if address_type == 0x01:
        return _recv_exact(sock, 4)
    if address_type == 0x03:
        length = _recv_exact(sock, 1)[0]
        return _recv_exact(sock, length)
    if address_type == 0x04:
        return _recv_exact(sock, 16)
    raise OSError("SOCKS5 proxy returned an invalid address type.")


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("Socket closed before enough data was received.")
        data += chunk
    return data


def _relay(client: socket.socket, upstream: socket.socket) -> None:
    client.settimeout(None)
    upstream.settimeout(None)
    sockets = [client, upstream]
    while True:
        readable, _, errored = select.select(sockets, [], sockets, SOCKET_TIMEOUT_SECONDS)
        if errored:
            return
        if not readable:
            continue
        for source in readable:
            destination = upstream if source is client else client
            try:
                data = source.recv(RELAY_BUFFER_SIZE)
                if not data:
                    return
                destination.sendall(data)
            except OSError:
                return
