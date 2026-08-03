from __future__ import annotations

import pytest

from coregulation_poc.capture.devices import (
    DeviceKind,
    parse_directshow_device_log,
    select_device,
)
from coregulation_poc.capture.directshow import build_directshow_input
from coregulation_poc.capture.media import MediaSourceError


def test_parse_directshow_devices_preserves_stable_indices() -> None:
    inventory = parse_directshow_device_log(
        [
            '"Integrated Camera" (video)',
            '  Alternative name "@device_pnp_camera"',
            '"USB Camera" (video)',
            '  Alternative name "@device_pnp_usb"',
            '"Microphone Array" (audio)',
            '  Alternative name "@device_cm_microphone"',
        ]
    )

    assert [device.index for device in inventory.cameras] == [0, 1]
    assert inventory.cameras[0].alternative_name == "@device_pnp_camera"
    assert inventory.microphones[0].kind is DeviceKind.MICROPHONE
    public = inventory.as_public_dict()
    assert "@device" not in str(public)


def test_select_device_requires_an_explicit_available_selector() -> None:
    inventory = parse_directshow_device_log(
        ['"Integrated Camera" (video)', '"Microphone Array" (audio)']
    )

    assert (
        select_device(inventory.cameras, kind=DeviceKind.CAMERA, index=0).name
        == "Integrated Camera"
    )
    with pytest.raises(ValueError, match="exactly one"):
        select_device(inventory.cameras, kind=DeviceKind.CAMERA)
    with pytest.raises(MediaSourceError, match="unavailable"):
        select_device(inventory.cameras, kind=DeviceKind.CAMERA, index=2)


def test_directshow_input_uses_selected_capture_names() -> None:
    inventory = parse_directshow_device_log(
        [
            '"Integrated Camera" (video)',
            '"Microphone Array" (audio)',
        ]
    )

    assert build_directshow_input(inventory.cameras[0], inventory.microphones[0]) == (
        'video="Integrated Camera":audio="Microphone Array"'
    )
