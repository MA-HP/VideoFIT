"""
Metrology Vision Pro — Application Settings Model
Dataclasses and JSON loading for appsettings.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class CameraInfo:
    """Represents a single camera entry from appsettings.json."""
    name: str = "Unknown"
    serial: str = ""
    config_file: str = ""
    calibration_px_mm: str = "0.0"


@dataclass
class AppDefaults:
    """Application-level default values."""
    comparison_mode: str = "Best Fit"


@dataclass
class AppSettings:
    """Root settings model loaded from appsettings.json."""
    app_defaults: AppDefaults = field(default_factory=AppDefaults)
    cameras: list[CameraInfo] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> AppSettings:
        """Load and parse appsettings.json from the given path."""
        if not os.path.exists(path):
            print(f"WARNING: appsettings.json not found at {path}")
            return cls()

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            defaults_raw = raw.get("app_defaults", {})
            app_defaults = AppDefaults(
                comparison_mode=defaults_raw.get("comparison_mode", "Best Fit"),
            )

            cameras = [
                CameraInfo(
                    name=c.get("name", "Unknown"),
                    serial=c.get("serial", ""),
                    config_file=c.get("config_file", ""),
                    calibration_px_mm=c.get("calibration_px_mm", "0.0"),
                )
                for c in raw.get("cameras", [])
            ]

            print(f"Loaded appsettings.json successfully. Found {len(cameras)} cameras.")
            return cls(app_defaults=app_defaults, cameras=cameras)

        except Exception as e:
            print(f"Failed to parse appsettings.json: {e}")
            return cls()

