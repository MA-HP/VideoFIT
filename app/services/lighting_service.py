"""
VideoFIT — Lighting Service
One socket per command, exactly like malight_poc.py — the device closes the
connection after each response so a persistent socket always gets dropped.
"""

from __future__ import annotations

import socket

CHANNEL_NAMES = {1: "EPI", 2: "-", 3: "COAX", 4: "DIA"}


class LightingService:

    def __init__(self, ip_address: str = "169.254.5.100", port: int = 62077) -> None:
        self.ip_address = ip_address
        self.port = port

    def set_intensity(self, channel: int, intensity: float) -> bool:
        intensity_str = f"{intensity:.2f}".rstrip("0").rstrip(".")
        return self._send(f"CH{channel},{intensity_str}\n")

    def set_on(self, channel: int) -> bool:
        return self._send(f"CH{channel},ON\n")

    def set_off(self, channel: int) -> bool:
        return self._send(f"CH{channel},OFF\n")

    def _send(self, command: str) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self.ip_address, self.port))
            if not command.endswith("\n"):
                command += "\n"
            sock.sendall(command.encode("ascii"))
            sock.recv(8192)          # drain the response
            sock.close()
            return True
        except Exception as e:
            print(f"[LightingService] {e}")
            return False
