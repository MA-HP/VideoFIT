"""
VideoFIT — Application Settings Model
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
    heatmap_min_error: float = 0.1   # mm — below this = fully green
    heatmap_max_error: float = 0.5   # mm — above this = fully red


@dataclass
class AppDefaults:
    """Application-level default values."""
    comparison_mode: str = "Best Fit"
    fit_objective: str = "Strict"
    heatmap_color_low: str = "#00FF00"   # color at heatmap_min (in-tolerance)
    heatmap_color_mid: str = "#FF8000"   # color at heatmap_max boundary
    heatmap_color_high: str = "#FF0000"  # color above heatmap_max (out-of-tolerance)


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
                fit_objective=defaults_raw.get("fit_objective", "Strict"),
                heatmap_color_low=defaults_raw.get("heatmap_color_low", "#00FF00"),
                heatmap_color_mid=defaults_raw.get("heatmap_color_mid", "#FF8000"),
                heatmap_color_high=defaults_raw.get("heatmap_color_high", "#FF0000"),
            )

            cameras = [
                CameraInfo(
                    name=c.get("name", "Unknown"),
                    serial=c.get("serial", ""),
                    config_file=c.get("config_file", ""),
                    calibration_px_mm=c.get("calibration_px_mm", "0.0"),
                    heatmap_min_error=float(c.get("heatmap_min_error", 1.0)),
                    heatmap_max_error=float(c.get("heatmap_max_error", 3.0)),
                )
                for c in raw.get("cameras", [])
            ]

            print(f"Loaded appsettings.json successfully. Found {len(cameras)} cameras.")
            return cls(app_defaults=app_defaults, cameras=cameras)

        except Exception as e:
            print(f"Failed to parse appsettings.json: {e}")
            return cls()

