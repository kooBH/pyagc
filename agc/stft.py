import numpy as np


def stft(x, frame_size, hop_size=None, window=None, N=None, only_positive_freqs=True):
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal x.

    x:          signal (for now, only mono is supported)
    frame_size: in samples
    hop_size:   in samples (default 25% of frame size)
    window:     numpy array with the window to be used
    N:          number of FFT points to compute
    only_positive_freqs:
                if True, only the positive FFT bins (including DC) are returned
    """

    # set defaults and sanity check
    assert type(frame_size) == int
    if hop_size is None:
        hop_size = frame_size / 4  # the default windows (stft & isftf) are designed to work with 25% overlap
    assert type(hop_size) == int
    if window is None:
        # window = 0.5 * (1. - np.cos(2. * np.pi * np.arange(frame_size) / frame_size))
        window = np.hanning(frame_size + 1)[:-1]
    assert window.size == frame_size
    if N is None:
        N = frame_size
    assert type(N) == int

    # compute the stft
    X = np.array([np.fft.fft(window * x[i:i + frame_size], N) for i in range(0, len(x) - frame_size + 1, hop_size)]).T

    # if requested, remove the "negative frequencies"
    if only_positive_freqs:
        X = X[:int(N / 2) + 1, :]

    return X


def istft(X, frame_size, hop_size=None, window=None, only_positive_freqs=True):
    """
    Compute the Inverse Short-Time Fourier Transform (ISTFT) of a stft X.

    X:          Complex STFT (columns encode time and rows encode frequency)
    frame_size: time-domain frame size to use, in sample (each column in X correspond to these many samples in time)
    hop_size:   in samples (default 25% of frame size)
    only_positive_freqs:
                if True, only the positive bins (including DC) are considered to be included in X
    """

    # set defaults and sanity check
    assert type(frame_size) == int
    if hop_size is None:
        hop_size = frame_size / 4  # the default windows (stft & isftf) are designed to work with 25% overlap
    assert type(hop_size) == int
    if window is None:
        # window = 0.5 * (1. - np.cos(2. * np.pi * np.arange(frame_size) / frame_size))
        window = np.hanning(frame_size + 1)[:-1]
        # make it COLA for 25% overlap when using the above window
        window = window * 2. / 3.
    assert window.size == frame_size

    # if required, construct the full spectrogram
    if only_positive_freqs:
        X = np.vstack((X, np.flipud(np.conj(X[1:-1, :]))))

    # allocate output array
    x = np.zeros(X.shape[1] * hop_size + frame_size - hop_size)

    # compute istft by: IFFT + OLA
    for n, i in enumerate(range(0, len(x) - frame_size + 1, hop_size)):
        x[i:i + frame_size] += window * np.real(np.fft.ifft(X[:, n], n=frame_size))

    return x


