import click
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import cmath

from copy import deepcopy
from typing import Sequence

from mmwave.radar import Radar
from mmwave.utils import Stopwatch

from processing import *
from config import radar_hardware_config, processing_config

SAVE_BACKGROUND = False

# For showing PDP of original or background subtracted
# profiles.


def custom_pause(interval):
    """
    Custom pause function that does not steal focus
    :param interval: pause duration in seconds
    """
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        fig_manager = matplotlib._pylab_helpers.Gcf.get_active()
        if fig_manager is not None:
            canvas = fig_manager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)


@click.command()
@click.argument("config_paths", type=click.Path(dir_okay=False), nargs=-1)
def main(config_paths: Sequence[str]):
    radar = Radar(pid=radar_hardware_config['pid'], vid=radar_hardware_config['vid'])
    print(f'main.py> radar found')

    for i, config_path in enumerate(config_paths):
        is_first = i == 0
        radar.configure(config_path, flush=is_first)

    # radar.cli.send_cmd("guiMonitor -1 0 1 0 0 0 1")

    print(f"FPS is: {radar.config.fps:.2f}")
    print(f"Radar cube shape is: {(radar.config.n_tx, 1, radar.config.n_rx, radar.config.n_range_bins, 2)}")
    print(f'Radar config: {vars(radar.config)}')

    # Data rate:
    # fps: 20, n_rx: 4
    estimated_data_rate = radar.config.fps * radar.config.n_rx * radar.config.n_range_bins * 4 * 8
    max_data_rate = radar.config.data_baud_rate
    print(f"Data rate is {estimated_data_rate:.0f}/{max_data_rate} bps "
          f"({100 * estimated_data_rate/max_data_rate:.2f}%)")

    radar.start()

    # Release CLI port as it's not needed anymore
    radar.close_cli()

    # Set up range graph
    plt.ion()
    n_range_bins = radar.config.n_range_bins
    fig = plt.figure("Range profile")  # type:plt.Figure
    axes = fig.add_subplot(1, 1, 1,
                           xlim=(int(-0.1 * n_range_bins), int(1.1 * n_range_bins)),
                           ylim=(-.1, 1))  # type:plt.Axes
    graph, = axes.plot(np.arange(n_range_bins), np.zeros(n_range_bins))  # type:plt.Line2D
    plt.show()
    plt.draw()

    sw = Stopwatch()
    sw.start()

    last_frames = []

    try:
        while True:
            if not plt.get_fignums():
                break

            packet = radar.read()
            sw.lap()

            radar_cube = packet.radar_cube
            if radar_cube is None:
                continue

            # print(f'shape: {radar_cube.shape}')

            # Select the first (and only) chirp
            radar_cube = radar_cube[:,0,:].astype(np.float)
            print("Before making one number", radar_cube.shape)
            # Cast as complex
            radar_cube = radar_cube[..., 0] + 1j * radar_cube[..., 1]
            print("After making one number", radar_cube.shape)

            # radar_cube_average = radar_cube.mean(axis=0).mean(axis=0)
            range_data_mag = np.abs(radar_cube).mean(axis=0).mean(axis=0)
            average_profile_norm = range_data_mag / range_data_mag.max()

            peaks  = find_peaks(average_profile_norm)

            phases = []
            max_bin = find_max_phase(peaks, range_data_mag)
            receivers = [0, 3]
            for i in range(3):
                for j in range(2):
                    print(cmath.phase(radar_cube[i,receivers[j],[max_bin]]))
                    phases.append(cmath.phase(radar_cube[i,receivers[j],[max_bin]]))
                
            for i in range(1, 7, 2):
                print("phase shift", phases[i] - phases[i-1])
            graph.set_ydata(average_profile_norm)

            plt.draw()
            custom_pause(1 / 1000)

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        radar.stop()

def find_max_phase(peaks,range_data_mag):
    maxPeak = range_data_mag[0]
    for i in range(len(peaks[0])):
        if range_data_mag[peaks[0][i]] > maxPeak:
            maxPeak = range_data_mag[peaks[0][i]]
            maxPeakIndex = peaks[0][i]
    return maxPeakIndex

if __name__ == "__main__":
    main()


