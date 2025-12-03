import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import firwin


def artifact_filter(
    image, artifact_frequency=0.25, taps=33, epsilon=0.04, print_params=False
):
    image = np.asarray(image, float)
    return image - lowpass(
        highpass(image, artifact_frequency, taps, epsilon, print_params),
        taps,
        epsilon,
        print_params,
    )


def highpass(image, distortion_freq, taps, epsilon, print_params=False):
    highpass_filter = firwin(
        taps, distortion_freq - epsilon, pass_zero="highpass", fs=1
    )
    if print_params:
        print("Highpass FIR Parameters:")
        print(highpass_filter)
    return convolve1d(image, highpass_filter, axis=0)


def lowpass(image, taps, epsilon, print_params=False):
    lowpass_filter = firwin(taps, epsilon, pass_zero="lowpass", fs=1)
    if print_params:
        print("Lowpass FIR Parameters:")
        print(lowpass_filter)
    return convolve1d(image, lowpass_filter, axis=1)
