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
from config import radar_hardware_config, processing_config, collection_config

FNAME = '1010'


@click.command()
@click.argument("config_paths", type=click.Path(dir_okay=False), nargs=-1)
def main(config_paths: Sequence[str]):
    radar = Radar(pid=radar_hardware_config['pid'], vid=radar_hardware_config['vid'])
    print(f'main.py> radar found')

    for i, config_path in enumerate(config_paths):
        is_first = i == 0
        radar.configure(config_path, flush=is_first)

    print(f"FPS is: {radar.config.fps:.2f}")
    print(f"Radar cube shape is: {(radar.config.n_tx, 1, radar.config.n_rx, radar.config.n_range_bins, 2)}")
    print(f'Radar config: {vars(radar.config)}')

    # Data rate:
    estimated_data_rate = radar.config.fps * radar.config.n_rx * radar.config.n_range_bins * 4 * 8
    max_data_rate = radar.config.data_baud_rate
    print(f"Data rate is {estimated_data_rate:.0f}/{max_data_rate} bps "
          f"({100 * estimated_data_rate/max_data_rate:.2f}%)")

    radar.start()
    # Release CLI port as it's not needed anymore
    radar.close_cli()

    radar_cube_collection = {}
    average_profile_collection = {}
    angle_indices = {}

    for trace in range(collection_config['traces']):
    # for trace in range(1):
        radar_cube_collection[trace] = []
        average_profile_collection[trace] = []
        angle_indices[trace] = []
        print(f'Press Enter to start trace {trace}.')
        input()

        curr_index = 0
        curr_angle = collection_config['start_angle']

        # Warm-up
        for idx in range(10):
            packet = radar.read()

        while True:
            print(f'angle: {curr_angle}')
            packet = radar.read()
            radar_cube = packet.radar_cube
            if radar_cube is None:
                continue

            # Select the first (and only) chirp
            radar_cube = radar_cube[:,0,:]
            # Cast as complex
            radar_cube = radar_cube[..., 0] + 1j * radar_cube[..., 1]
            average_profile, peaks = get_reflections(radar_cube)

            radar_cube_collection[trace].append(radar_cube)
            average_profile_collection[trace].append(average_profile)

            # Need minimum of 3 measurements to compute diffs
            if len(average_profile_collection[trace]) < 3:
                continue

            curr_diff = np.mean(np.abs(average_profile_collection[trace][-1] - average_profile_collection[trace][-2]))
            last_diff = np.mean(np.abs(average_profile_collection[trace][-2] - average_profile_collection[trace][-3]))

            if (last_diff - curr_diff) > collection_config['change_thresh']:
                if len(angle_indices[trace]) == 0:
                    curr_angle += 1
                    angle_indices[trace].append(curr_index)
                else:
                    # Consecutive angles must be atleast 3 frames (i.e., 600ms apart)
                    if curr_index - angle_indices[trace][-1] > 3:
                        curr_angle += 1
                        angle_indices[trace].append(curr_index)

            if curr_angle > collection_config['end_angle']:
                break

            curr_index += 1

    # mag_diffs = []
    # for idx in range(len(average_profile_collection[0]) - 1):
    #     diff = np.mean(np.abs(average_profile_collection[0][idx+1] - average_profile_collection[0][idx]))
    #     mag_diffs.append(diff)

    # plt.plot(mag_diffs)
    # plt.show()

    print('Saving data.')
    with open(f'data/{FNAME}.pickle', 'wb') as fh:
        pickle.dump({'radar_cube': radar_cube_collection,
                     'average_profile': average_profile_collection,
                     'angle_indices': angle_indices}, fh)
    print('Saved.')

if __name__ == "__main__":
    main()
