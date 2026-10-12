"""Passive listener evidence for the configured inference endpoint."""
from __future__ import annotations

import ipaddress
import os
import subprocess
from urllib.parse import urlsplit


def endpoint_host(host):
    return urlsplit(host).hostname if host.startswith(("ws://", "wss://")) else host.strip("[]")


def loopback(host):
    host = endpoint_host(host)
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def listener_matches(endpoint, host, port):
    """Match address family as well as port; don't guess IPv6 dual-stack mode."""
    try:
        address, bound_port = endpoint.rsplit(":", 1)
        if int(bound_port) != port:
            return False
        bound = ipaddress.ip_address(address.strip("[]"))
        host = endpoint_host(host)
        targets = [ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1")] if host.lower() == "localhost" else [ipaddress.ip_address(host)]
        return any(bound.version == target.version and (bound.is_unspecified or bound == target) for target in targets)
    except ValueError:
        return False


def local_listener(host, port):
    """True/False for loopback listeners, None if inspection isn't available."""
    if not loopback(host):
        return None
    windows = os.name == "nt"
    command = ["netstat.exe", "-ano", "-p", "tcp"] if windows else ["ss", "-H", "-ltn"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=.75,
                                **({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if windows else {}))
        if result.returncode:
            return None
        for line in result.stdout.splitlines():
            fields = line.split()
            if windows:
                if len(fields) >= 4 and fields[0].upper() == "TCP" and fields[3].upper() == "LISTENING":
                    if listener_matches(fields[1], host, port):
                        return True
            elif len(fields) >= 4 and fields[0] == "LISTEN":
                if listener_matches(fields[3], host, port):
                    return True
        return False
    except (OSError, subprocess.SubprocessError):
        return None
