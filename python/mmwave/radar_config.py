from typing import Optional, Sequence
from .cli.cmds import ChannelCfg, FrameCfg, ProfileCfg, ConfigDataPort


class RadarConfig:
    # channelCfg
    n_tx: int = None
    n_rx: int = None

    # profileCfg
    n_range_bins: int = None

    # frameCfg
    n_chirps: int = None
    fps: float = None

    data_baud_rate: int = None

    def __init__(self, cmds: Sequence = None):
        cmds = cmds or []

        self.n_tx: Optional[int] = None
        self.n_rx: Optional[int] = None
        self.n_range_bins: Optional[int] = None
        self.n_chirps: Optional[int] = None
        self.fps: Optional[float] = None

        self.data_baud_rate: Optional[int] = None

        for cmd in cmds:
            self.update(cmd)

    def update(self, cmd):
        if isinstance(cmd, ChannelCfg):
            self.n_tx = bin(int(cmd.tx_mask)).count("1")
            self.n_rx = bin(int(cmd.rx_mask)).count("1")
        elif isinstance(cmd, ProfileCfg):
            self.n_range_bins = cmd.adc_samples
        elif isinstance(cmd, FrameCfg):
            self.n_chirps = cmd.loops
            self.fps = 1000 / cmd.periodicity
        elif isinstance(cmd, ConfigDataPort):
            self.data_baud_rate = cmd.data_rate
