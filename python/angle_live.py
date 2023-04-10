import click
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

from copy import deepcopy
from typing import Sequence

from mmwave.radar import Radar
from mmwave.utils import Stopwatch

from processing import *
from config import radar_hardware_config, processing_config


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

            print(f'shape: {radar_cube.shape}')

            # Select the first (and only) chirp
            radar_cube = radar_cube[:,0,:].astype(np.float)
            # Cast as complex
            radar_cube = radar_cube[..., 0] + 1j * radar_cube[..., 1]
            average_profile, peaks = get_reflections(radar_cube)

            # Calculate angles
            angles = get_angles(radar_cube, peaks[0])

            for peak in angles:
                print(peak, angles[peak])

            average_profile_norm = average_profile / average_profile.max()

            # graph.set_ydata(subtracted_profile_norm)
            graph.set_ydata(average_profile_norm)

            plt.draw()
            custom_pause(1 / 1000)

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        radar.stop()


if __name__ == "__main__":
    main()
