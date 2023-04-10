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


            # x = list(range(0, 256))
            # # Plotting for all of the antennas 
            # for i in range(3):
            #     for j in range(4):
            #         plt.plot(x, np.abs(radar_cube[i][j]/np.abs(radar_cube[i][j].max())))
            #         # custom_pause(1 / 1000)


            # Mean across all antennas
            radar_cube_average = radar_cube.mean(axis=0).mean(axis=0)
            range_data_mag = np.abs(radar_cube).mean(axis=0).mean(axis=0)

            average_profile_norm = range_data_mag / range_data_mag.max()

            # print(f'shape of average profile: {average_profile_norm.shape}')
            # print(f'first entry of average profile: {average_profile_norm[0]}')
            # print(f'first entry of average profile: {average_profile_norm}')



            # Finding peaks
            peaks = find_peaks(average_profile_norm)
            print(f'these are peaks: {peaks}')
            print(f'number of peaks: {len(peaks[0])}')
            
            # # max_peak = peaks[0][0][0]

            # index = peaks[0][0]
            # print(len(peaks[0]))
            # print(max_peak_1)
            

            # finding phase for TX 3 and RX 1
            print(f'finding phase for TX 3 and RX 1')
            max_peak_1 = radar_cube[2,0,[peaks[0][0]]]
            for i in range(len(peaks[0])):
                if radar_cube[2,0,[peaks[0][i]]] > max_peak_1:
                    max_peak_1 = radar_cube[2][0][peaks[0][i]]
                    index = peaks[0][i]
            print (f'max peak: {max_peak_1}')
            print (f'max peak at bin number: {index}')
            phase_1 = cmath.phase(max_peak_1)
            print (f'phase: {phase_1}')
            # print (f'numpy phase: {np.angle(max_peak)}')


            # finding phase for TX 3 and RX 4
            max_peak_2 = radar_cube[2,3,[peaks[0][0]]]
            print(f'finding phase for TX 3 and RX 4')
            for i in range(len(peaks[0])):
                if radar_cube[2,3,[peaks[0][i]]] > max_peak_2:
                    max_peak_2 = radar_cube[2][3][peaks[0][i]]
                    index = peaks[0][i]
            print (f'max peak: {max_peak_2}')
            print (f'max peak at bin number: {index}')
            phase_2 = cmath.phase(max_peak_2)
            print (f'phase: {phase_2}')

            phase_shift_1 = phase_1 - phase_2
            print (f'phase shift: {phase_shift_1}')


            # finding phase for TX 2 and RX 1
            max_peak_3 = radar_cube[1,0,[peaks[0][0]]]
            print(f'finding phase for TX 2 and RX 1')
            for i in range(len(peaks[0])):
                if radar_cube[1,0,[peaks[0][i]]] > max_peak_3:
                    max_peak_3 = radar_cube[1,0,[peaks[0][i]]]
                    index = peaks[0][i]
            print (f'max peak: {max_peak_3}')
            print (f'max peak at bin number: {index}')
            phase_3 = cmath.phase(max_peak_3)
            print (f'phase: {phase_3}')

            # finding phase for TX 2 and RX 4
            max_peak_4 = radar_cube[1,3,[peaks[0][0]]]
            print(f'finding phase for TX 2 and RX 4')
            for i in range(len(peaks[0])):
                if radar_cube[1,3,[peaks[0][i]]] > max_peak_4:
                    max_peak_4 = radar_cube[1,3,[peaks[0][i]]]
                    index = peaks[0][i]
            print (f'max peak: {max_peak_4}')
            print (f'max peak at bin number: {index}')
            phase_4 = cmath.phase(max_peak_4)
            print (f'phase: {phase_4}')

            phase_shift_2 = phase_3 - phase_4
            print (f'next phase shift: {phase_shift_2}')

            # finding phase for TX 1 and RX 1
            max_peak_5 = radar_cube[0,0,[peaks[0][0]]]
            print(f'finding phase for TX 1 and RX 1')
            for i in range(len(peaks[0])):
                if radar_cube[0,0,[peaks[0][i]]] > max_peak_5:
                    max_peak_4 = radar_cube[0,0,[peaks[0][i]]]
                    index = peaks[0][i]
            print (f'max peak: {max_peak_5}')
            print (f'max peak at bin number: {index}')
            phase_5 = cmath.phase(max_peak_5)
            print (f'phase: {phase_5}')

            # finding phase for TX 1 and RX 4
            max_peak_6 = radar_cube[0,3,[peaks[0][0]]]
            print(f'finding phase for TX 1 and RX 1')
            for i in range(len(peaks[0])):
                if radar_cube[0,3,[peaks[0][i]]] > max_peak_6:
                    max_peak_6 = radar_cube[0,3,[peaks[0][i]]]
                    index = peaks[0][i]
            print (f'max peak: {max_peak_6}')
            print (f'max peak at bin number: {index}')
            phase_6 = cmath.phase(max_peak_6)
            print (f'phase: {phase_6}')

            phase_shift_3 = phase_5 - phase_6
            print (f'next to next phase shift: {phase_shift_3}')
            
            graph.set_ydata(average_profile_norm)

            plt.draw()
            custom_pause(1 / 1000)

            if SAVE_BACKGROUND:
                with open("pickles/background.pickle", "wb") as fh:
                    pickle.dump(average_profile_norm, fh)

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        radar.stop()
def max_peak(peaks,range_data_mag):
    maxPeak = range_data_mag[0]
    for i in range(len(peaks[0])):
        if range_data_mag[peaks[0][i]] > maxPeak:
            maxPeak = range_data_mag[peaks[0][i]]
            maxPeakIndex = peaks[0][i]
    return maxPeakIndex

if __name__ == "__main__":
    main()