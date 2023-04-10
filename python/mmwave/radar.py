import serial
import numpy as np
from typing import Optional, Sequence, Union

from .cli import MMWaveCLI, parse_config, ConfigCmds
from .device import Device, find_device, KNOWN_DEV_TYPES
from .packet import MsgHeader, TLVHeader, TimingInfo, TemperatureStats, UartPacket
from .radar_config import RadarConfig
from .utils import parse_cstruct

# Magic word at the beginning of every UART package
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"


class Radar:
    def __init__(self, vid: int = None, pid: int = None, open_cli: bool = True):
        if vid is not None and pid is not None:
            self.device = find_device(vid=vid, pid=pid)
        else:
            # Autodiscover device
            devs = (find_device(dev_type=dev_type) for dev_type in KNOWN_DEV_TYPES.values())
            self.device = next((d for d in devs if d is not None), None)

        assert self.device is not None, "Couldn't find device"

        # TODO: check using queryDemoStatus
        self.is_started = False

        # TODO optional: query full config (e.g. queryDemoParams)
        self.config: Optional[RadarConfig] = None

        self.cli: Optional[MMWaveCLI] = None
        self.data_stream: Optional[serial.Serial] = None

        if open_cli:
            self.open_cli()

        # TODO syntax sugar: allow config from constructor

    # region Control

    def open_cli(self, flush=True):
        self.cli = MMWaveCLI(self.device.user_port, flush=flush)

    def close_cli(self):
        self.cli.close()
        self.cli = None

    def _check_cli(self):
        assert self.cli is not None, "CLI needs to be open to send user commands"

    def start(self, listen: bool = True):
        self._check_cli()
        self.cli.send_cmd("sensorStart")
        self.is_started = True

        if listen:
            self.open_data_port()

    def stop(self):
        self._check_cli()
        self.cli.send_cmd("sensorStop")
        self.is_started = False

    def configure(self, config: Union[RadarConfig, ConfigCmds], flush: bool = True):
        print(f'radar.py> entering configure.')
        assert not flush or not self.is_started
        self._check_cli()
        print(f'radar.py> checked cli.')

        if flush:
            # TODO: remove when is_started becomes reliable
            self.cli.send_cmd("sensorStop")
            self.cli.send_cmd("flushCfg")

        print(f'radar.py> flushed.')

        if isinstance(config, RadarConfig):
            self.config = config
            raise NotImplementedError("Missing conversion from RadarConfig to cfg commands")
        else:
            cmds = parse_config(config, cfg_only=True)

            # Intercept configuration
            self.config = self.config or RadarConfig()
            for cmd in cmds:
                print(f'radar.py> cmd: {cmd}')
                self.config.update(cmd)

        # Execute parsed cfg commands through CLI
        for cmd in cmds:
            self.cli.send_cmd(cmd)

    # endregion

    # region Data

    def open_data_port(self, flush=True):
        if self.data_stream is not None:
            return

        self.data_stream = serial.Serial(self.device.data_port, self.config.data_baud_rate or 921600)
        assert self.data_stream is not None, "Unable to connect to data port"

        if flush:
            if self.data_stream.isOpen():
                self.data_stream.close()
                self.data_stream.open()

    def close_data_port(self):
        self.data_stream.close()
        self.data_stream = None

    def read(self):
        assert self.data_stream is not None
        buff = self.data_stream.read_until(MAGIC_WORD)
        magic_word = buff[-len(MAGIC_WORD):]
        assert (magic_word == MAGIC_WORD)

        header = parse_cstruct(MsgHeader, self.data_stream)
        packet = UartPacket(header)

        for i in range(header.num_tlvs):
            tlv_header = parse_cstruct(TLVHeader, self.data_stream)
            # print(tlv_header)
            if tlv_header.type == 2:
                tlv_data = self.data_stream.read(tlv_header.length)
                shape = (self.config.n_tx, 1, self.config.n_rx, self.config.n_range_bins, 2)
                # shape = (self.config.n_tx, self.config.n_chirps, self.config.n_rx, self.config.n_range_bins, 2)
                packet.radar_cube = np.frombuffer(tlv_data, dtype=np.int16).reshape(shape)
                break
            elif tlv_header.type == 6:
                packet.timing_info = parse_cstruct(TimingInfo, self.data_stream)
            elif tlv_header.type == 9:
                packet.temp_stats = parse_cstruct(TemperatureStats, self.data_stream)
            else:
                # Just skip the unsupported TLV. Using read as seek is not supported by serial.Serial
                self.data_stream.read(tlv_header.length)

        return packet

    # endregion
