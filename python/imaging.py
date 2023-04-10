import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from copy import deepcopy

from config import collection_config
from processing import *

FNAME = '1010'
NUM_TRACES = 3
angle_thresh = .1

with open(f'data/{FNAME}.pickle', 'rb') as fh:
    data = pickle.load(fh)

def plot_shape(cube, rbin):
    plt.plot(cube[rbin, :])


def old_imaging():
    radar_cube_collection = data['radar_cube']
    average_profile_collection = data['average_profile']
    angle_indices_collection = data['angle_indices']

    for trace in range(NUM_TRACES):
        sliced_average_profiles = []
        angle_profiles = []

        for idx in range(len(average_profile_collection[trace])): # 1 2 3 .. 26 .. 155
            if idx in angle_indices_collection[trace]: # 26, 32, 36 ... 155
                radar_cube = radar_cube_collection[trace][idx+1]
                _, peaks = get_reflections(radar_cube)
                peaks = peaks[0]
                # Returns angles (Azimuth AoA, Azimuth AoD, Elevation AoA and Elevation AoD) of all peaks
                # Angles is a dictionary where each key is a peak index, and corresponding value is a list of angles
                # {2: [0.1, 0.2, 0.4, 0.3], 7: [ ... ], ...}

                angles = get_angles(radar_cube, peaks)

                # profile_angles = get_angles(radar_cube, range(0,35))
                # profile_angles_list = [profile_angles[peak][2] for peak in range(0,35)]
                # angle_profiles.append(profile_angles_list)

                # average_profile = np.abs(radar_cube_collection[trace][idx+1][1,2,:])

                average_profile = average_profile_collection[trace][idx+1]
                average_profile = average_profile / average_profile.max()

                # --- Extrapolation ---
                N = 256
                print(radar_cube.shape)
                new_radar_cube = np.zeros((radar_cube.shape[0], radar_cube.shape[1], N)) + \
                                 1j * np.zeros((radar_cube.shape[0], radar_cube.shape[1], N))
                for tx in range(radar_cube.shape[0]):
                    for rx in range(radar_cube.shape[1]):
                        ifft = np.fft.ifft(radar_cube[tx,rx,:])
                        new_radar_cube[tx,rx,:] = np.fft.fft(ifft, n=N)

                average_profile = np.abs(new_radar_cube).mean(axis=0).mean(axis=0)
                average_profile = average_profile / average_profile.max()

                # --- END Extrapolation

                # plt.figure()
                # plt.plot(average_profile)
                # plt.title(peaks)
                # print(peaks)
                # plt.show()

                # new_average_profile = []
                # for idx in range(N):
                #     # Checks (i) if index is a peak, (ii) small AoA, (iii) small AoD
                #     # if idx in angles and np.abs(angles[idx][2]) < angle_thresh and np.abs(angles[idx][3]) < angle_thresh:
                #     # if idx in angles and np.abs(angles[idx][0]) < angle_thresh:
                #     # if idx in angles:
                #         new_average_profile.append(average_profile[idx])
                #     else:
                #         new_average_profile.append(0)

                # This will plot the raw spectrogram
                sliced_average_profiles.append(average_profile)
                # This will plot the cleaned up spectrogram
                # sliced_average_profiles.append(new_average_profile)


        sliced_average_profiles = np.array(sliced_average_profiles)
        print(f'shape: {sliced_average_profiles.shape}')

        # angle_profiles = np.array(angle_profiles)
        # print(f'shape: {angle_profiles.shape}')

        # for rbin in range(10, 16):
        #     plt.figure()
        #     plot_shape(sliced_average_profiles, rbin=rbin)
        #     plt.title(f'{rbin} Power')

        # for rbin in range(5, 9):
        #     plt.figure()
        #     plot_shape(angle_profiles, rbin=rbin)
        #     plt.title(f'{rbin} AOA')

        # Pick first 35 range bins
        # sliced_average_profiles = sliced_average_profiles[:, 60:130].T
        sliced_average_profiles = sliced_average_profiles[:, 0:40].T
        # sliced_average_profiles = sliced_average_profiles[:, 10:60].T

        plt.figure()
        plt.imshow(sliced_average_profiles)
        plt.title('Power')
        plt.xlabel('Angle (degree); 0 = -10')
        plt.ylabel('Distance (1bin = 4.4cm)')

        # # Pick first 35 range bins
        # angle_profiles = angle_profiles[:, 0:35].T
        # plt.figure()
        # plt.imshow(angle_profiles)
        # plt.title('AOAs')
        # # plt.show()

    plt.show()

def imaging():
    radar_cube_collection = data['radar_cube']
    average_profile_collection = data['average_profile']
    angle_indices_collection = data['angle_indices']

    for trace in range(NUM_TRACES):
        sliced_average_profiles = []
        angle_profiles = []

        for idx in range(len(average_profile_collection[trace])): # 1 2 3 .. 26 .. 155
            if idx in angle_indices_collection[trace]: # 26, 32, 36 ... 155
                radar_cube = radar_cube_collection[trace][idx+1]

                # --- Extrapolation ---
                N = 256
                print(radar_cube.shape)
                new_radar_cube = np.zeros((radar_cube.shape[0], radar_cube.shape[1], N)) + \
                                 1j * np.zeros((radar_cube.shape[0], radar_cube.shape[1], N))
                for tx in range(radar_cube.shape[0]):
                    for rx in range(radar_cube.shape[1]):
                        ifft = np.fft.ifft(radar_cube[tx,rx,:])
                        new_radar_cube[tx,rx,:] = np.fft.fft(ifft, n=N)

                average_profile = np.abs(new_radar_cube).mean(axis=0).mean(axis=0)[:]
                average_profile = average_profile / average_profile.max()
                # --- END Extrapolation

                # --- Peaks ---
                peaks = find_peaks(average_profile)[0]
                # new_average_profile = []
                # for idx in range(len(average_profile)):
                #     if idx not in peaks:
                #         new_average_profile.append(0)
                #     else:
                #         new_average_profile.append(average_profile[idx])

                # average_profile = new_average_profile
                # --- END Peaks ---

                # --- Angles ---
                angles = get_angles(radar_cube, peaks)

                # Azimuth AOA has a negative bias: at center, it's around -0.5

                new_average_profile = []
                for idx in range(256):
                    if idx in angles and angles[idx][0] < 0.25 and angles[idx][0] > -1.5 and angles[idx][1] < 0.05 and angles[idx][1] > -0.15:
                        new_average_profile.append(average_profile[idx])
                    else:
                        new_average_profile.append(0)

                average_profile = new_average_profile

                # --- END Angles ---
                # This will plot the raw spectrogram
                sliced_average_profiles.append(average_profile)
                # This will plot the cleaned up spectrogram
                # sliced_average_profiles.append(new_average_profile)


        sliced_average_profiles = np.array(sliced_average_profiles)
        print(f'shape: {sliced_average_profiles.shape}')


        sliced_average_profiles = sliced_average_profiles[:, 0:25].T
        # sliced_average_profiles = sliced_average_profiles[:, 10:60].T

        # for rbin in range(3, 9):
        #     plt.figure()
        #     plot_shape(sliced_average_profiles, rbin=rbin)
        #     plt.title(f'{rbin} Power')

        plt.figure()
        plt.imshow(sliced_average_profiles)
        plt.title('Power')
        plt.xlabel('Angle (degree); 0 = -10')
        plt.ylabel('Distance (1bin = 4.4cm)')

    plt.show()

# Print peaks and angles to see actual values
# Modify plots to overlay red squares where direct reflections are
#

def pdp_plots():
    radar_cube_collection = data['radar_cube']
    average_profile_collection = data['average_profile']
    angle_indices_collection = data['angle_indices']

    trace = 0

    for idx in range(len(average_profile_collection[trace])):
        if idx in angle_indices_collection[trace]:
            radar_cube = radar_cube_collection[trace][idx+1]
            _, peaks = get_reflections(radar_cube)
            peaks = peaks[0]

            average_profile = average_profile_collection[trace][idx+1]
            normalized_average_profile = average_profile / average_profile.max()

            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(average_profile[:50])
            plt.subplot(2,1,2)
            plt.plot(normalized_average_profile[:50])
            plt.title(idx)

            plt.show()



if __name__ == '__main__':
    imaging()
    # pdp_plots()