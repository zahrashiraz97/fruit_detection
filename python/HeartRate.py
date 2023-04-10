
import click
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import cmath
import math

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

    # # Set up range graph
    # plt.ion()
    # n_range_bins = radar.config.n_range_bins
    # fig = plt.figure("Range profile")  # type:plt.Figure
    # axes = fig.add_subplot(1, 1, 1,
    #                        xlim=(int(-0.1 * n_range_bins), int(1.1 * n_range_bins)),
    #                        ylim=(-.1, 1))  # type:plt.Axes
    # graph, = axes.plot(np.arange(n_range_bins), np.zeros(n_range_bins))  # type:plt.Line2D
    # plt.show()
    # plt.draw()

    # Set up range graph for heart rate
    plt.ion()
    n_range_bins = radar.config.n_range_bins
    fig = plt.figure("Range profile")  # type:plt.Figure
    axes = fig.add_subplot(1, 1, 1,
                           xlim=(int(-0.1 * n_range_bins), int(1.1 * n_range_bins)),
                           ylim=(-1, 1))  # type:plt.Axes
    graph, = axes.plot(np.arange(n_range_bins), np.zeros(n_range_bins))  # type:plt.Line2D
    plt.show()
    plt.draw()

    sw = Stopwatch()
    sw.start()

    consecutive_frames = []
    heart_rate_sequence = [0] * 256
    index = 0
    static_constant = 10
    mean_background = np.zeros(256)

    try:
        # each while loop is for one frame 
        # 20 times in one second 
        # one radar cube is one frame 
        while True:
            if not plt.get_fignums():
                break

            packet = radar.read()
            sw.lap()
            

            radar_cube = packet.radar_cube
            if radar_cube is None:
                continue

            # Select the first (and only) chirp
            radar_cube = radar_cube[:,0,:].astype(np.float)
            # print("Before making one number", radar_cube.shape)
            # Cast as complex
            radar_cube = radar_cube[..., 0] + 1j * radar_cube[..., 1]
            # print("After making one number", radar_cube.shape)
            # Mean across all antennas
            radar_cube_average = radar_cube.mean(axis=0).mean(axis=0)
            range_data_mag = np.abs((radar_cube).mean(axis=0).mean(axis=0))

            average_profile_norm = range_data_mag / range_data_mag.max()
            consecutive_frames.append(average_profile_norm)
            index += 1


            if index >= static_constant:
                # Take the average of all the frames in the list 
                mean_background = np.mean(consecutive_frames[index - static_constant:index], axis = 0)
            subtracted_range_profile = np. abs(average_profile_norm - mean_background)            

            # Finding peaks
            peaks = find_peaks(subtracted_range_profile)
            # print(f'these are peaks: {peaks}')
            # print(f'number of peaks: {len(peaks[0])}')

            # print(f'Max peak at index number:{max_peak(peaks,range_data_mag)}')

            maximumpeak, maxpeaksecond = max_peak(peaks,subtracted_range_profile)
            print(f'Max peak at index number:{maximumpeak}')
            if index >= static_constant:
                heart_rate = (cmath.phase(radar_cube_average[maximumpeak])) / (subtracted_range_profile[maximumpeak])
                heart_rate_sequence.insert(index - 1, heart_rate) 
                print("heart rate:", heart_rate)
                average_heart_rate_norm = [x / max(heart_rate_sequence) for x in heart_rate_sequence]  
                graph.set_ydata(average_heart_rate_norm[max(0, index - 256):max(index, 256)])
                plt.draw()
            


            # # # Finding Distance
            distance = maximumpeak * 3.75
            print(f'Distance from object in centimeters:{distance}') 

            # # # Finding AoD
            phase_one = cmath.phase(radar_cube[0, 0, maximumpeak])
            phase_two = cmath.phase(radar_cube[1, 0, maximumpeak])
            unwrap = np.unwrap([phase_one, phase_two])
            phase_shift = unwrap[0] - unwrap[1]
            # print("AoD phase shift for the first peak", phase_shift)
            AoD_first= np.degrees(np.arcsin(phase_shift/(2*math.pi)))
            # print(f'Angle of departure in degrees for the first one: {AoD_first}')

            receivers = [0, 3]
            phase_shifts = []
            phase_shifts_second = []

            for i in range(3):
                phases = []
                phases_second = []
                for j in range(2):
                    # print(cmath.phase(radar_cube[i,receivers[j], maximumpeak]))
                    phases.append(cmath.phase(radar_cube[i,receivers[j],maximumpeak]))
                    # print(cmath.phase(radar_cube[i,receivers[j], maxpeaksecond]))
                    phases_second.append(cmath.phase(radar_cube[i,receivers[j],maxpeaksecond]))
                unwrap = np.unwrap([phases[1], phases[0]])
                unwrap_second = np.unwrap([phases_second[1], phases_second[0]])
                phase_shifts.append(unwrap[0] - unwrap[1])
                phase_shifts_second.append(unwrap_second[1] - unwrap_second[0])

            AoA_first= np.degrees(np.arcsin(phase_shifts[0]/math.pi))
            AoA_second= np.degrees(np.arcsin(phase_shifts_second[0]/math.pi))
            print(f'Angle of arrival in degrees for the first one: {AoA_first}')
            # print(f'Angle of arrival in degrees for the second peak: {AoA_second}')


            # Subtracted background range profile
            # graph.set_ydata(subtracted_range_profile)
            # plt.draw()
            
            custom_pause(1 / 1000)

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        radar.stop()

def max_peak(peaks,range_data_mag):
    temp = []

    for i in range(len(peaks[0])):
        temp.append(range_data_mag[peaks[0][i]])
    temp = sorted(temp, reverse=True)
    max_result = np.where(range_data_mag == temp[0])[0][0]
    max_peak_second = np.where(range_data_mag == temp[1])[0][0]


    return max_result, max_peak_second




if __name__ == "__main__":
    main()

