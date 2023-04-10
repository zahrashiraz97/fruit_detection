
import click
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import cmath
import math
from scipy.fft import fft, fftfreq, rfft, rfftfreq
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
    # n_range_bins = radar.config.n_range_bins
    n_range_bins = 300
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
    phases_breathing = []
    mags_breathing = []

    index = 0
    static_constant = 10
    mean_background = np.zeros(256)

    subtracted_range_profile_average = []
    radar_cube_list = []

    wait_at_start = 40
    wait_at_start_counter = 0

    n_frames = n_range_bins - static_constant


    try:
        # each while loop is for one frame 
        # 20 times in one second 
        # one radar cube is one frame 
        while True:
            if not plt.get_fignums():
                break

            packet = radar.read()
            sw.lap()

            if wait_at_start_counter < wait_at_start:
                print(f'Waiting: counter = {wait_at_start_counter}')
                wait_at_start_counter += 1
                continue

            

            radar_cube = packet.radar_cube
            if radar_cube is None:
                continue

            # Select the first (and only) chirp
            radar_cube = radar_cube[:,0,:].astype(np.float)

            # Cast as complex
            radar_cube = radar_cube[..., 0] + 1j * radar_cube[..., 1]

            # Mean across all antennas
            radar_cube_average = radar_cube.mean(axis=0).mean(axis=0)
            range_data_mag = np.abs((radar_cube).mean(axis=0).mean(axis=0))

            average_profile_norm = range_data_mag / range_data_mag.max()
            consecutive_frames.append(average_profile_norm)
            radar_cube_list.append(radar_cube)
            index += 1

            if index >= static_constant:
                # Take the average of all the frames in the list 
                mean_background = np.mean(consecutive_frames[index - static_constant:index], axis = 0)

            subtracted_range_profile = np.abs(average_profile_norm - mean_background)
            subtracted_range_profile_average.append([subtracted_range_profile])

            if index == n_range_bins:
                np_form = np.array(subtracted_range_profile_average)
                subtracted_range_profile_max_bin = np_form.mean(axis=0)
                give_it_to_max = np.array(subtracted_range_profile_max_bin[0])
                
                 # Finding peaks
                peaks = find_peaks(give_it_to_max)
                maximumpeak, maxpeaksecond = max_peak(peaks, give_it_to_max)

                for j in range(index):
                    values = cmath.polar(radar_cube_list[j][0,3,maximumpeak])    
                    mags_breathing.append(values[0]) 
                    phases_breathing.append(values[1]) 

                norm = [float(i)/max(phases_breathing) for i in phases_breathing]
                with open('data_23F.pickle', 'wb') as fh:
                    pickle.dump({'radar_cube_list': radar_cube_list, 'max_bin': maximumpeak,
                                 'mags_breathing': mags_breathing, 'phases_breathing': phases_breathing,
                                'bkg_subtracted': subtracted_range_profile_average}, fh)


                fft_mag = np.abs(np.fft.fft(mags_breathing[static_constant:]))
                fft_pha = np.abs(np.fft.fft(phases_breathing[static_constant:]))

                # Find peaks in fft_mag
                peaks_fft_mag = find_peaks(fft_mag)
                breath, heart = max_peak(peaks_fft_mag, fft_mag)
                print(fft_mag[breath],fft_mag[heart])
                print('Using Mag: ', fft_mag[breath] * 60, fft_mag[heart] * 60)
                
                # Find peaks in fft_pha
                # peaks_fft_pha = find_peaks(fft_pha)
                # breath_pha, heart_pha = max_peak(peaks_fft_pha, fft_pha)
                # print('Using pha: ', breath_pha * 60, heart_pha * 60) 

                print(maximumpeak)
                plt.figure(figsize=(12,8))
                plt.subplot(2,2,1)
                plt.plot(mags_breathing[static_constant:]) ## this 10 is because we are finding fft after 10 frames because in first 10 frames subtracted profile is being calculated
                plt.xlabel('Mag')
                plt.subplot(2,2,3)
                plt.plot(np.unwrap(phases_breathing[static_constant:]))
                plt.xlabel('Phase')
                plt.subplot(2,2,2)
                xf = np.fft.rfftfreq(n_frames, 1 / 5)
                plt.plot(xf[2:], fft_mag[2:int(n_frames/2 + 1)])
                plt.xlabel('Mag FFT')
                plt.subplot(2,2,4)
                plt.plot(xf[2:], fft_pha[2:int(n_frames/2 + 1)])
                plt.xlabel('Pha FFT')

                # graph.set_ydata(norm)
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

