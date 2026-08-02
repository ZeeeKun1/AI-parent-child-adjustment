"""Shared WebSocket transport helper.

``websocket-client`` resolves hostnames with ``AF_UNSPEC`` and iterates the
returned addresses. On networks where IPv6 routes are broken, the IPv6
``connect`` call hangs until the socket timeout, and the library re-raises
``socket.timeout`` instead of falling through to the next IPv4 address.

This module resolves the hostname to IPv4 only, opens the TCP connection
manually, wraps it with SSL when needed, and hands the pre-connected socket
to ``websocket.create_connection`` via its ``socket`` option.
"""

from __future__ import annotations

import socket
import ssl
from urllib.parse import urlparse

import websocket


def create_websocket_connection(
    url: str,
    *,
    header: list[str],
    timeout: float,
) -> websocket.WebSocket:
    """Open a WebSocket connection using IPv4 only.

    Parameters
    ----------
    url:
        Full WebSocket URL (``wss://`` or ``ws://``).
    header:
        List of ``"Key: Value"`` HTTP headers for the handshake.
    timeout:
        Socket timeout in seconds, applied to both TCP connect and the
        WebSocket receive path.

    Returns
    -------
    websocket.WebSocket
        A connected, handshaked WebSocket ready for ``send`` / ``recv``.

    Raises
    ------
    ConnectionError
        If DNS resolution, TCP connect, SSL wrap, or the WebSocket
        handshake fails.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        raise ConnectionError(f"Cannot parse hostname from URL: {url}")
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    is_secure = parsed.scheme == "wss"

    try:
        addrinfo_list = socket.getaddrinfo(
            host, port, socket.AF_INET, socket.SOCK_STREAM, socket.SOL_TCP
        )
    except socket.gaierror as exc:
        raise ConnectionError(f"IPv4 DNS resolution failed for {host}: {exc}") from exc

    if not addrinfo_list:
        raise ConnectionError(f"No IPv4 address found for {host}")

    last_error: Exception | None = None
    for addrinfo in addrinfo_list:
        family, socktype, proto = addrinfo[:3]
        sockaddr = addrinfo[4]
        raw_sock = socket.socket(family, socktype, proto)
        raw_sock.settimeout(timeout)
        try:
            raw_sock.connect(sockaddr)
        except OSError as exc:
            raw_sock.close()
            last_error = exc
            continue

        try:
            sock: socket.socket = raw_sock
            if is_secure:
                ssl_context = ssl.create_default_context()
                sock = ssl_context.wrap_socket(raw_sock, server_hostname=host)

            return websocket.create_connection(
                url,
                header=header,
                timeout=timeout,
                socket=sock,
            )
        except Exception as exc:
            sock.close()
            last_error = exc
            continue

    raise ConnectionError(
        f"All IPv4 connection attempts to {host}:{port} failed: {last_error}"
    )
