from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import av

from coregulation_poc.capture.media import MediaSourceError

DEVICE_LINE = re.compile(r'"(?P<name>.+)"\s+\((?P<kind>video|audio)\)', re.IGNORECASE)
ALTERNATIVE_LINE = re.compile(r'Alternative name\s+"(?P<name>.+)"', re.IGNORECASE)


class DeviceKind(StrEnum):
    CAMERA = "camera"
    MICROPHONE = "microphone"


@dataclass(frozen=True, slots=True)
class MediaDevice:
    kind: DeviceKind
    index: int
    name: str
    alternative_name: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("device index must be non-negative")
        if not self.name.strip():
            raise ValueError("device name must not be empty")

    @property
    def capture_name(self) -> str:
        return self.alternative_name or self.name

    def manifest_record(self) -> dict[str, object]:
        fingerprint_source = self.alternative_name or self.name
        fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:12]
        return {
            "kind": self.kind.value,
            "index": self.index,
            "display_name": self.name,
            "identifier_fingerprint": fingerprint,
            "raw_hardware_identifier_saved": False,
        }


@dataclass(frozen=True, slots=True)
class DeviceInventory:
    cameras: tuple[MediaDevice, ...]
    microphones: tuple[MediaDevice, ...]

    def __post_init__(self) -> None:
        if any(device.kind is not DeviceKind.CAMERA for device in self.cameras):
            raise ValueError("camera inventory contains a non-camera device")
        if any(
            device.kind is not DeviceKind.MICROPHONE for device in self.microphones
        ):
            raise ValueError("microphone inventory contains a non-microphone device")

    def as_public_dict(self) -> dict[str, object]:
        return {
            "backend": "windows_directshow",
            "cameras": [device.manifest_record() for device in self.cameras],
            "microphones": [device.manifest_record() for device in self.microphones],
        }


def parse_directshow_device_log(messages: Iterable[str]) -> DeviceInventory:
    """Parse FFmpeg DirectShow device-list messages without exposing raw logs."""
    parsed: list[tuple[DeviceKind, str, str | None]] = []
    for message in messages:
        device_match = DEVICE_LINE.search(message)
        if device_match:
            kind = (
                DeviceKind.CAMERA
                if device_match.group("kind").lower() == "video"
                else DeviceKind.MICROPHONE
            )
            parsed.append((kind, device_match.group("name"), None))
            continue
        alternative_match = ALTERNATIVE_LINE.search(message)
        if alternative_match and parsed:
            kind, name, alternative = parsed[-1]
            if alternative is None:
                parsed[-1] = (kind, name, alternative_match.group("name"))

    cameras: list[MediaDevice] = []
    microphones: list[MediaDevice] = []
    for kind, name, alternative_name in parsed:
        target = cameras if kind is DeviceKind.CAMERA else microphones
        target.append(
            MediaDevice(
                kind=kind,
                index=len(target),
                name=name,
                alternative_name=alternative_name,
            )
        )
    return DeviceInventory(cameras=tuple(cameras), microphones=tuple(microphones))


def list_windows_media_devices() -> DeviceInventory:
    """Enumerate DirectShow camera and microphone endpoints on Windows."""
    if os.name != "nt":
        raise MediaSourceError("Live device enumeration currently requires Windows.")
    previous_level = av.logging.get_level()
    logs: list[object]
    container = None
    try:
        av.logging.set_level(av.logging.INFO)
        with av.logging.Capture(local=False) as captured:
            try:
                container = av.open(
                    "dummy",
                    format="dshow",
                    options={"list_devices": "true"},
                )
            except av.FFmpegError:
                pass
        logs = list(captured)
    finally:
        if container is not None:
            container.close()
        av.logging.set_level(previous_level)

    message_fragments: list[str] = []
    for item in logs:
        if hasattr(item, "message"):
            message_fragments.append(str(item.message))
        elif isinstance(item, tuple) and len(item) >= 3:
            message_fragments.append(str(item[2]))
        else:
            message_fragments.append(str(item))
    inventory = parse_directshow_device_log("".join(message_fragments).splitlines())
    if not inventory.cameras and not inventory.microphones:
        raise MediaSourceError(
            "DirectShow returned no camera or microphone devices. "
            "Check Windows privacy permissions and device connections."
        )
    return inventory


def select_device(
    devices: tuple[MediaDevice, ...],
    *,
    kind: DeviceKind,
    index: int | None = None,
    name: str | None = None,
) -> MediaDevice:
    if (index is None) == (name is None):
        raise ValueError(f"Select exactly one {kind.value} by index or name.")
    if index is not None:
        if index < 0 or index >= len(devices):
            raise MediaSourceError(
                f"{kind.value} index {index} is unavailable; found {len(devices)} device(s)."
            )
        return devices[index]
    normalized_name = (name or "").casefold()
    matches = [device for device in devices if device.name.casefold() == normalized_name]
    if len(matches) != 1:
        raise MediaSourceError(
            f"Could not uniquely match {kind.value} name {name!r}; "
            f"available names: {[device.name for device in devices]}"
        )
    return matches[0]
