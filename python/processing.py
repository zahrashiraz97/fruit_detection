import numpy as np
from scipy.signal import find_peaks

# azim_aoa_pairs: [TX1-RX1, TX1-RX4], [TX2-RX1, TX2-RX4], [TX3-RX1, TX3-RX4]
#                 (180 degree shift), [TX1-RX2, TX1-RX3], [TX2-RX1, TX2-RX3], [TX3-RX2, TX3-RX3]
azim_aoa_pairs = [[(0, 0), (0, 3)], [(1, 0), (1, 3)], [(2, 0), (2, 3)]]

# azim_aod_pairs (lambda spacing): [TX1-RX1, TX2-RX1], [TX1-RX4, TX2-RX4]
azim_aod_pairs = [[(0, 0), (1, 0)], [(0, 3), (1, 3)]]

# elev_aoa_pairs (180 degree shift): [TX1-RX1, TX1-RX2], [TX1-RX4, TX1-RX3], [TX2-RX1, TX2-RX2],
#                                    [TX2-RX4, TX2-RX3], [TX3-RX1, TX3-RX2], [TX3-RX4, TX3-RX3]
elev_aoa_pairs = [[(0, 0), (0, 1)], [(0, 3), (0, 2)], [(1, 0), (1, 1)], [(1, 3), (1, 2)], [(2, 0), (2, 1)], [(2, 3), (2, 2)]]

# elev_aod_pairs: [TX2-RX1, TX3-RX1], [TX2-RX4, TX3-RX4]
elev_aod_pairs = [[(1, 0), (2, 0)], [(1, 3), (2, 3)]]


def get_reflections(radar_cube):
    """
    Returns:
        Averaged range profile and peaks (reflections)
    """

    # Mean across TX and RX axes
    range_data_mag = np.abs(radar_cube).mean(axis=0).mean(axis=0)
    # DEBUG: Across one antenna only
    # range_data_mag = np.abs(radar_cube[1,1,:])
    peaks = find_peaks(range_data_mag)

    return range_data_mag, peaks

def get_angles(radar_cube, peaks):

    angles = {}
    for peak in peaks:

        if peak > 30: break

        radar_cube_single_bin = radar_cube[..., peak]
        azim_aoa = calculate_azim_aoa(radar_cube_single_bin)
        azim_aod = calculate_azim_aod(radar_cube_single_bin)
        elev_aoa = calculate_elev_aoa(radar_cube_single_bin)
        elev_aod = calculate_elev_aod(radar_cube_single_bin)
        angles[peak] = (azim_aoa, azim_aod, elev_aoa, elev_aod)
        # print(peak, azim_aoa, azim_aod, elev_aoa, elev_aod)

    return angles

def calculate_azim_aoa(radar_cube_single_bin):

    azim_aoas = []
    for (pair1, pair2) in azim_aoa_pairs:
        angle_1 = np.angle(radar_cube_single_bin[pair1[0], pair1[1]])
        angle_2 = np.angle(radar_cube_single_bin[pair2[0], pair2[1]])
        diff = np.angle(np.exp(1j*angle_1) / np.exp(1j*angle_2))
        azim_aoas.append(diff)

    azim_aoa = np.median(azim_aoas)
    return azim_aoa

def calculate_azim_aod(radar_cube_single_bin):

    azim_aods = []
    for (pair1, pair2) in azim_aod_pairs:
        angle_1 = np.angle(radar_cube_single_bin[pair1[0], pair1[1]])
        angle_2 = np.angle(radar_cube_single_bin[pair2[0], pair2[1]])
        diff = np.angle(np.exp(1j*angle_1) / np.exp(1j*angle_2))
        # Dividing by 2 because of full wavelength
        diff = diff / 2
        azim_aods.append(diff)

    azim_aod = np.median(azim_aods)
    return azim_aod

def calculate_elev_aoa(radar_cube_single_bin):

    # Need 180 degree phase shift

    elev_aoas = []
    for (pair1, pair2) in elev_aoa_pairs:
        angle_1 = np.angle(radar_cube_single_bin[pair1[0], pair1[1]])
        angle_2 = np.angle(radar_cube_single_bin[pair2[0], pair2[1]])
        diff = np.angle(np.exp(1j*angle_1) / np.exp(1j*angle_2) * np.exp(1j*0.5*np.pi))
        elev_aoas.append(diff)

    elev_aoa = np.median(elev_aoas)
    return elev_aoa

def calculate_elev_aod(radar_cube_single_bin):

    elev_aods = []
    for (pair1, pair2) in elev_aod_pairs:
        angle_1 = np.angle(radar_cube_single_bin[pair1[0], pair1[1]])
        angle_2 = np.angle(radar_cube_single_bin[pair2[0], pair2[1]])
        diff = np.angle(np.exp(1j*angle_1) / np.exp(1j*angle_2))
        elev_aods.append(diff)

    elev_aod = np.median(elev_aods)
    return elev_aod
