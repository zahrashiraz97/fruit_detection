from .cmd import mmwavecli_cfg, mmwavecli_cmd, RegisteredCmd


@mmwavecli_cmd("sensorStop")
class SensorStop(RegisteredCmd):
    pass


@mmwavecli_cmd("flushCfg")
class FlushCfg(RegisteredCmd):
    pass


@mmwavecli_cmd("sensorStart")
class SensorStart(RegisteredCmd):
    pass


@mmwavecli_cfg('dfeDataOutputMode')
class DfeDataOutputMode(RegisteredCmd):
    type: int


@mmwavecli_cfg('channelCfg')
class ChannelCfg(RegisteredCmd):
    rx_mask: int
    tx_mask: int
    cascading: int


@mmwavecli_cfg('adcCfg')
class AdcCfg(RegisteredCmd):
    adc_bits: int
    output_format: int


@mmwavecli_cfg('adcbufCfg')
class AdcBufCfg(RegisteredCmd):
    subframe_idx: int
    output_format: int
    sample_swap: int
    channel_interleave: int
    chirp_threshold: int


@mmwavecli_cfg('profileCfg')
class ProfileCfg(RegisteredCmd):
    id: int
    start_freq: float
    idle_time: float
    adc_start_time: float
    ramp_end_time: float
    tx_out_power: int
    tx_phase_shifter: int
    freq_slope: float
    tx_start_time: float
    adc_samples: int
    sample_rate: int
    hpf_corner_freq1: int
    hpf_corner_freq2: int
    rx_gain: int


@mmwavecli_cfg('chirpCfg')
class ChirpCfg(RegisteredCmd):
    start_idx: int
    end_idx: int
    profile: int
    var_start_freq: float
    var_freq_slope: float
    var_idle_time: float
    var_adc_start_time: float
    tx_mask: int


@mmwavecli_cfg('frameCfg')
class FrameCfg(RegisteredCmd):
    start_idx: int
    end_idx: int
    loops: int
    frames: int
    periodicity: float
    trigger: int
    trigger_delay: float


@mmwavecli_cfg('lowPower')
class LowerPower(RegisteredCmd):
    chain: int
    adc_mode: int


@mmwavecli_cfg('configDataPort')
class ConfigDataPort(RegisteredCmd):
    data_rate: int
    ack: int
