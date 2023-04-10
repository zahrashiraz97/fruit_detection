import numpy as np
from typing import NamedTuple, Optional
from .utils import cstruct


@cstruct("8I")
class MsgHeader(NamedTuple):
    version: int
    total_packet_len: int
    platform: int
    frame_number: int
    time_cpu_cycles: int
    num_detected_obj: int
    num_tlvs: int
    subframe_num: int


@cstruct("2I")
class TLVHeader(NamedTuple):
    type: int
    length: int


@cstruct("6I")
class TimingInfo(NamedTuple):
    inter_frame_processing_time: int
    transmit_output_time: int
    inter_frame_processing_margin: int
    inter_chirp_processing_margin: int
    active_frame_cpu_load: int
    inter_frame_cpu_load: int


@cstruct("iI10h")
class TemperatureStats(NamedTuple):
    temp_report_valid: int
    time: int
    tmp_rx0_sens: int
    tmp_rx1_sens: int
    tmp_rx2_sens: int
    tmp_rx3_sens: int

    tmp_tx0_sens: int
    tmp_tx1_sens: int
    tmp_tx2_sens: int

    tmp_pm_sens: int
    tmp_dig0_sens: int
    tmp_dig1_sens: int


class UartPacket:
    header: MsgHeader
    radar_cube: Optional[np.ndarray] = None
    timing_info: Optional[TimingInfo] = None
    temp_stats: Optional[TemperatureStats] = None

    def __init__(self, header: MsgHeader):
        self.header = header
