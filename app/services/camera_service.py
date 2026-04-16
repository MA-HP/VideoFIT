"""
VideoFIT — Camera Service
Business logic for camera lifecycle: enumeration, connection, streaming.
The CameraManager model (structure/state) lives in app.models.camera.
"""

from __future__ import annotations

import json
import os

import cv2
import imagingcontrol4 as ic4
from PySide6.QtCore import QObject, Signal


class FrameSignaler(QObject):
    """Thread-safe bridge: emits a Qt signal carrying an OpenCV image (numpy array)."""
    frame_ready = Signal(object)


class _SinkListener(ic4.QueueSinkListener):
    """IC4 sink listener that converts incoming buffers to RGB and forwards them."""

    def __init__(self, callback):
        self._callback = callback

    def sink_connected(self, sink, image_type, min_buffers_required) -> bool:
        return True

    def frames_queued(self, sink):
        buffer = sink.pop_output_buffer()
        buffer_wrap = buffer.numpy_wrap()
        rgb_img = cv2.cvtColor(buffer_wrap, cv2.COLOR_BGR2RGB)
        self._callback(rgb_img)


class CameraService(QObject):
    """
    Manages camera hardware lifecycle using IC4.

    Responsibilities
    ----------------
    - Device enumeration
    - Opening / closing a device by serial number
    - Applying a JSON config file to the device property map
    - Starting / stopping the stream

    Signals
    -------
    frame_ready(numpy.ndarray)
        Emitted for every new frame (RGB).
    """

    frame_ready = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._grabber = ic4.Grabber()
        self._signaler = FrameSignaler()
        self._signaler.frame_ready.connect(self.frame_ready)
        self._listener = _SinkListener(self._signaler.frame_ready.emit)
        self._sink = ic4.QueueSink(
            self._listener, [ic4.PixelFormat.BGR8], max_output_buffers=1
        )
        self._connected_devices: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_device_list(self) -> list:
        """Re-enumerate physically connected cameras and return the list."""
        self._connected_devices = list(ic4.DeviceEnum.devices())
        return self._connected_devices

    @property
    def is_streaming(self) -> bool:
        return self._grabber.is_device_open

    def open_camera(self, serial: str, config_path: str | None = None) -> None:
        """
        Open the camera identified by *serial*, optionally apply a config file,
        and start streaming.

        Raises
        ------
        RuntimeError
            If the camera serial is not found among connected devices.
        """
        self.disconnect()

        if not self._connected_devices:
            self.refresh_device_list()

        device = next(
            (d for d in self._connected_devices if d.serial == serial), None
        )
        if device is None:
            raise RuntimeError(f"Camera serial {serial} is not connected.")

        self._grabber.device_open(device)

        if config_path:
            self._apply_config(config_path)

        self._grabber.stream_setup(self._sink)
        print(f"Started camera: {device.model_name}")

    def disconnect(self) -> None:
        """Stop streaming and close the device (safe to call even if not open)."""
        if self._grabber.is_device_open:
            self._grabber.stream_stop()
            self._grabber.device_close()

    def release(self) -> None:
        """
        Explicitly release all IC4 resources (Grabber, Sink, DeviceInfo list).
        Must be called before ic4.Library.exit() to avoid destructor errors.
        """
        self.disconnect()
        self._connected_devices.clear()
        # Drop references so IC4 objects are destroyed NOW while library is still alive
        self._sink = None  # type: ignore[assignment]
        self._grabber = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_config(self, path: str) -> None:
        """Push an IC4 JSON config file into the currently-opened device."""
        if not self._grabber.is_device_open:
            return

        if not os.path.exists(path):
            print(f"WARNING: Camera config file missing: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            hardware_props = config_data.get("properties", config_data)
            clean_json = json.dumps(hardware_props)
            prop_bytes = bytearray(clean_json.encode("utf-8"))

            try:
                self._grabber.device_property_map.deserialize(prop_bytes)
                print(f"Successfully applied hardware params: {os.path.basename(path)}")
            except ic4.IC4Exception as e:
                if getattr(e, "code", None) == 34 or "Incomplete" in str(e):
                    print(
                        f"Applied {os.path.basename(path)} with warnings: "
                        "Some locked properties were ignored."
                    )
                else:
                    raise

        except Exception as e:
            print(f"Error applying configuration from {path}: {e}")

