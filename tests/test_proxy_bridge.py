from __future__ import annotations

import socket
import threading

from googleaisearch2api.proxy_bridge import LocalSocksProxyBridge, build_socks_proxy_target


class FakeSocks5Server:
    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.host, self.port = self._listener.getsockname()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._done = threading.Event()
        self.error: BaseException | None = None
        self.methods: bytes = b""
        self.username: str | None = None
        self.password: str | None = None
        self.destination_type: int | None = None
        self.destination_host: str | None = None
        self.destination_port: int | None = None
        self.tunneled_payload: bytes = b""

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._listener.close()
        self._thread.join(timeout=5)

    def wait(self) -> None:
        assert self._done.wait(5)
        if self.error is not None:
            raise AssertionError("Fake SOCKS5 server failed.") from self.error

    def _serve(self) -> None:
        try:
            conn, _ = self._listener.accept()
            with conn:
                conn.settimeout(5)
                version, method_count = _recv_exact(conn, 2)
                assert version == 0x05
                self.methods = _recv_exact(conn, method_count)
                assert 0x02 in self.methods
                conn.sendall(b"\x05\x02")

                auth_version, username_length = _recv_exact(conn, 2)
                assert auth_version == 0x01
                self.username = _recv_exact(conn, username_length).decode()
                password_length = _recv_exact(conn, 1)[0]
                self.password = _recv_exact(conn, password_length).decode()
                conn.sendall(b"\x01\x00")

                request_version, command, reserved, address_type = _recv_exact(conn, 4)
                assert request_version == 0x05
                assert command == 0x01
                assert reserved == 0x00
                self.destination_type = address_type
                if address_type == 0x03:
                    host_length = _recv_exact(conn, 1)[0]
                    self.destination_host = _recv_exact(conn, host_length).decode()
                elif address_type == 0x01:
                    self.destination_host = socket.inet_ntop(
                        socket.AF_INET,
                        _recv_exact(conn, 4),
                    )
                elif address_type == 0x04:
                    self.destination_host = socket.inet_ntop(
                        socket.AF_INET6,
                        _recv_exact(conn, 16),
                    )
                else:
                    raise AssertionError(f"Unexpected SOCKS5 address type: {address_type}")
                self.destination_port = int.from_bytes(_recv_exact(conn, 2), "big")
                conn.sendall(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")

                self.tunneled_payload = _recv_exact(conn, 4)
                conn.sendall(b"pong")
        except BaseException as exc:
            self.error = exc
        finally:
            self._done.set()


def test_local_socks_proxy_bridge_authenticates_and_tunnels_connect() -> None:
    fake_server = FakeSocks5Server()
    fake_server.start()
    target = build_socks_proxy_target(
        f"socks5h://{fake_server.host}:{fake_server.port}",
        username="openai",
        password="secret",
    )
    bridge = LocalSocksProxyBridge(target)
    bridge.start()
    try:
        bridge_host, bridge_port = bridge.server_url.removeprefix("http://").split(":")
        with socket.create_connection((bridge_host, int(bridge_port)), timeout=5) as client:
            client.settimeout(5)
            client.sendall(
                b"CONNECT example.com:443 HTTP/1.1\r\n"
                b"Host: example.com:443\r\n"
                b"\r\n"
            )
            response = client.recv(4096)
            assert response.startswith(b"HTTP/1.1 200 Connection Established")
            client.sendall(b"ping")
            assert _recv_exact(client, 4) == b"pong"
    finally:
        bridge.stop()
        fake_server.close()

    fake_server.wait()
    assert fake_server.username == "openai"
    assert fake_server.password == "secret"
    assert fake_server.destination_type == 0x03
    assert fake_server.destination_host == "example.com"
    assert fake_server.destination_port == 443
    assert fake_server.tunneled_payload == b"ping"


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("Socket closed before enough test data was received.")
        data += chunk
    return data
