"""Egress policy for the networked sandbox (ADR-007).

``EgressPolicy.validate`` resolves a target URL **once**, requires every
resulting address to be globally routable and returns an ``EgressTarget`` that
pins one of them. The sandbox never resolves or dials the target itself: a
forwarding container dials only the pinned address, so a hostname that later
re-resolves to a private address (DNS rebinding) is irrelevant. The hostname
survives only as a DNS alias inside the sandbox network, which keeps TLS SNI
and certificate-hostname verification working end to end.

The policy is injectable (resolver and address predicate) so tests can point it
at a synthetic target without relaxing the default policy.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from urllib.parse import urlsplit

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

_DEFAULT_PORTS = {"http": 80, "https": 443}
# Names become Docker network aliases, so keep to plain lowercase DNS syntax.
_DNS_NAME = re.compile(r"^(?=.{1,253}$)[a-z0-9]([a-z0-9.-]*[a-z0-9])?$")
# Global-looking ranges that still embed or tunnel to arbitrary IPv4 addresses.
_EMBEDDING_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),  # NAT64 well-known prefix
    ipaddress.ip_network("64:ff9b:1::/48"),  # NAT64 local-use prefix
    ipaddress.ip_network("2002::/16"),  # 6to4
    ipaddress.ip_network("2001::/32"),  # Teredo
)


class EgressDeniedError(PermissionError):
    """The target is not an allowed external destination."""


@dataclass(frozen=True, slots=True)
class EgressTarget:
    """A validated destination: ``address`` is the only place traffic may go."""

    scheme: Literal["http", "https"]
    hostname: str
    address: str
    port: int

    @property
    def is_ip_literal(self) -> bool:
        try:
            ipaddress.ip_address(self.hostname)
        except ValueError:
            return False
        return True


def normalize_address(raw: str) -> IPAddress:
    """Parse an address (dropping an IPv6 zone) and unwrap IPv4-mapped IPv6."""
    ip = ipaddress.ip_address(raw.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def is_public_address(ip: IPAddress) -> bool:
    """Default address policy: globally routable and not a tunnel/embedding range."""
    if not ip.is_global or ip.is_multicast:
        return False
    return not (
        isinstance(ip, ipaddress.IPv6Address)
        and any(ip in network for network in _EMBEDDING_NETWORKS)
    )


def system_resolver(hostname: str) -> list[str]:
    """Resolve *hostname* to every TCP address it maps to."""
    infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    return sorted({str(info[4][0]) for info in infos})


class EgressPolicy:
    """Validate target URLs and pin the address that may be dialled."""

    def __init__(
        self,
        resolver: Callable[[str], Sequence[str]] = system_resolver,
        address_allowed: Callable[[IPAddress], bool] = is_public_address,
    ) -> None:
        self._resolver = resolver
        self._address_allowed = address_allowed

    def validate(self, url: str) -> EgressTarget:
        """Return the pinned target for *url* or raise :class:`EgressDeniedError`."""
        parts = urlsplit(url.strip())
        if parts.scheme not in _DEFAULT_PORTS or not parts.hostname:
            msg = f"Target endpoint blocked: only http(s) URLs are allowed: {url}"
            raise EgressDeniedError(msg)
        if parts.username or parts.password:
            msg = f"Target endpoint blocked: credentials in URL: {url}"
            raise EgressDeniedError(msg)
        try:
            port = parts.port or _DEFAULT_PORTS[parts.scheme]
        except ValueError as exc:
            msg = f"Target endpoint blocked: invalid port: {url}"
            raise EgressDeniedError(msg) from exc
        if not 0 < port < 65536:
            msg = f"Target endpoint blocked: invalid port: {url}"
            raise EgressDeniedError(msg)

        try:
            addresses = self._resolver(parts.hostname)
        except (OSError, UnicodeError) as exc:
            msg = f"Target endpoint blocked: cannot resolve host: {url}"
            raise EgressDeniedError(msg) from exc
        if not addresses:
            msg = f"Target endpoint blocked: host has no addresses: {url}"
            raise EgressDeniedError(msg)

        resolved: list[IPAddress] = []
        for raw in addresses:
            try:
                ip = normalize_address(raw)
            except ValueError as exc:
                msg = f"Target endpoint blocked: unusable address for host: {url}"
                raise EgressDeniedError(msg) from exc
            if not self._address_allowed(ip):
                msg = f"Target endpoint blocked (SSRF prevention): {url}"
                raise EgressDeniedError(msg)
            resolved.append(ip)

        # Deterministic pin; IPv4 first because the egress bridge is IPv4-only.
        pinned = sorted(resolved, key=lambda ip: (ip.version, int(ip)))[0]
        scheme: Literal["http", "https"] = "https" if parts.scheme == "https" else "http"
        return EgressTarget(
            scheme=scheme,
            hostname=parts.hostname.lower(),
            address=str(pinned),
            port=port,
        )


def alias_for(target: EgressTarget) -> str:
    """DNS alias under which the sandbox reaches the forwarder.

    Raises:
        EgressDeniedError: The host cannot be expressed as a DNS alias (IP literal
            or non-ASCII/odd syntax); the sandbox network has no route to
            arbitrary addresses, so such targets are refused explicitly.
    """
    if target.is_ip_literal:
        msg = (
            "Target endpoint blocked: IP-literal targets are not supported by the "
            "sandbox egress proxy; use a DNS name"
        )
        raise EgressDeniedError(msg)
    try:
        name = target.hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        msg = "Target endpoint blocked: hostname cannot be expressed as an ASCII DNS name"
        raise EgressDeniedError(msg) from exc
    if not _DNS_NAME.fullmatch(name):
        msg = "Target endpoint blocked: hostname is not a plain DNS name"
        raise EgressDeniedError(msg)
    return name


# Source of the forwarding container. It is a plain L4 relay: it never parses
# the stream, never terminates TLS and dials only the address received on argv
# (already validated by EgressPolicy on the host side), so non-cooperating
# sandbox code cannot steer it anywhere else.
FORWARDER_SOURCE = """\
import socket, sys, threading

addr, port = sys.argv[1], int(sys.argv[2])
MAX_CONNECTIONS = 64
IDLE_SECONDS = 60
slots = threading.BoundedSemaphore(MAX_CONNECTIONS)


def pipe(src, dst):
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def serve(client):
    try:
        client.settimeout(IDLE_SECONDS)
        upstream = socket.create_connection((addr, port), timeout=10)
        upstream.settimeout(IDLE_SECONDS)
    except OSError:
        client.close()
        slots.release()
        return
    worker = threading.Thread(target=pipe, args=(client, upstream), daemon=True)
    worker.start()
    pipe(upstream, client)
    worker.join()
    client.close()
    upstream.close()
    slots.release()


listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("0.0.0.0", port))
listener.listen(64)
print("READY", flush=True)
while True:
    client, _ = listener.accept()
    if not slots.acquire(blocking=False):
        client.close()
        continue
    threading.Thread(target=serve, args=(client,), daemon=True).start()
"""
