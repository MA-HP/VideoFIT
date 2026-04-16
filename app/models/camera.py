"""
VideoFIT — Camera Model
Pure data structure representing the state of a camera connection.
All hardware logic (IC4 grabber, sink, streaming) lives in
app.services.camera_service.CameraService.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Camera:
    """Snapshot of the current camera connection state."""
    serial: str = ""
    model_name: str = ""
    is_streaming: bool = False
    config_path: str | None = None
    connected_serials: list[str] = field(default_factory=list)
