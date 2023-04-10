from serial.tools.list_ports import comports
from typing import NamedTuple, Optional
import usb.core
import usb.util


class DeviceType(NamedTuple):
    vid: int
    pid: int
    max_baud_rate: int = 921600


MMWAVEBOOST = DeviceType(0x0451, 0xbef3, max_baud_rate=3125000)
IWR6843ISK_ODS = DeviceType(0x10c4, 0xea70)

KNOWN_DEV_TYPES = {
    "MMWAVEBOOST": MMWAVEBOOST,
    "IWR6843ISK-ODS": IWR6843ISK_ODS
}


class Device(NamedTuple):
    user_port: str
    data_port: str
    max_baud_rate: int = 921600


def find_device(vid: int = None, pid: int = None, dev_type: DeviceType = None) -> Optional[Device]:
    dev_kwargs = {}

    if dev_type is not None:
        vid = dev_type.vid
        pid = dev_type.pid
    else:
        for name, dev_type in KNOWN_DEV_TYPES.items():
            if dev_type.vid == vid and dev_type.pid == pid:
                dev_kwargs['max_baud_rate'] = dev_type.max_baud_rate
                break

    """
    usb_dev = next((d for d in usb.core.find(find_all=True) if d.idVendor == vid and d.idProduct == pid), None)
    if usb_dev is None:
        return None

    sid = usb.util.get_string(usb_dev, usb_dev.iSerialNumber)
    ports = [p.device for p in comports() if p.vid == vid and p.pid == pid and p.serial_number == sid]
    assert len(ports) == 2, "Unknown device configuration detected"
    """

    data_port='/dev/cu.SLAB_USBtoUART3'
    user_port='/dev/cu.SLAB_USBtoUART'

    return Device(data_port=data_port, user_port=user_port, **dev_kwargs)
