import numpy as np
import soundfile as sf
from scipy.fftpack import fft, ifft
import numpy.matlib as npm
from scipy import signal as sg
from typing import Tuple, Optional, List

def stab(mat: np.ndarray, theta: float, num_channels: int) -> np.ndarray:
    """
    Stabilizes a matrix by adding a scaled identity matrix if its condition number is too high.

    This is often used to stabilize the spatial correlation matrix before inversion
    in beamforming algorithms like MVDR.

    Args:
        mat (np.ndarray): The complex square matrix to stabilize [num_channels x num_channels].
        theta (float): The condition number threshold. If cond(mat) > theta, stabilization is applied.
        num_channels (int): The number of channels (dimension of the matrix).

    Returns:
        np.ndarray: The stabilized matrix.
    """
    # Create a vector of decreasing powers of 10 for stabilization factor
    # Ensure dtype matches the input matrix for addition
    d = np.power(10.0, np.arange(-num_channels, 0, dtype=float)) # Use float for power calculation
    d = d.astype(mat.dtype) # Cast to matrix dtype (e.g., complex64)

    result_mat = mat.copy() # Work on a copy to avoid modifying the original matrix
    identity_mat = np.eye(num_channels, dtype=mat.dtype)

    # Calculate initial condition number
    try:
        cond_num = np.linalg.cond(result_mat)
    except np.linalg.LinAlgError:
        # Handle cases where the matrix might be singular initially
        cond_num = np.inf

    # Iteratively add scaled identity matrix until condition number is below threshold
    for i in range(num_channels):
        if cond_num <= theta:
            break # Matrix is stable enough
        # Add diagonal loading
        result_mat += d[i] * identity_mat
        # Recalculate condition number
        try:
            cond_num = np.linalg.cond(result_mat)
        except np.linalg.LinAlgError:
            cond_num = np.inf # Matrix became singular during stabilization attempt

    # If the loop finished without stabilizing below theta,
    # it might return the last stabilized version or the original
    # if theta was never exceeded. Consider adding a warning if stabilization fails.
    if cond_num > theta:
         print(f"Warning: Matrix stabilization might not have fully succeeded. Final condition number: {cond_num}")

    return result_mat

def get_3dim_spectrum(
    wav_name_template: str,
    channel_vec: List[int],
    start_point: int,
    stop_point: int,
    frame_len: int,
    shift_len: int,
    fftl: int
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Reads segments from multiple WAV files (one per channel) and computes their STFT.

    Args:
        wav_name_template (str): Template for WAV file names (e.g., 'channel_{}.wav'). '{}' is replaced by channel number.
        channel_vec (List[int]): List of channel numbers to read.
        start_point (int): Starting sample index to read from each file.
        stop_point (int): Ending sample index to read from each file.
        frame_len (int): Length of each STFT frame (window length).
        shift_len (int): Hop size (shift) between consecutive STFT frames.
        fftl (int): FFT length.

    Returns:
        Tuple[Optional[np.ndarray], Optional[int]]:
            - Spectrums (np.ndarray | None): 3D array [channel x frame x bin] containing the complex STFT spectrums, or None if reading fails.
            - num_samples (int | None): Number of samples read from the first channel file, or None if reading fails.
    """
    try:
        # Read the first channel to get the number of samples
        first_wav_name = wav_name_template.format(channel_vec[0])
        samples, sample_rate = sf.read(first_wav_name, start=start_point, stop=stop_point, dtype='float32')
        num_samples = len(samples)
        if num_samples == 0:
            print(f"Warning: No samples read from {first_wav_name} between {start_point} and {stop_point}")
            return None, None

        # Initialize array to hold waveform data for all channels
        dump_wav = np.zeros((len(channel_vec), num_samples), dtype=np.float32) # Use float32 for consistency
        dump_wav[0, :] = samples

        # Read remaining channels
        for i, channel_num in enumerate(channel_vec[1:], 1):
            wav_name = wav_name_template.format(channel_num)
            samples_ch, _ = sf.read(wav_name, start=start_point, stop=stop_point, dtype='float32')
            # Ensure all channels have the same length (important for consistency)
            if len(samples_ch) != num_samples:
                 print(f"Warning: Channel {channel_num} has different length ({len(samples_ch)}) than channel {channel_vec[0]} ({num_samples}). Truncating/Padding might occur implicitly later.")
                 # Handle length mismatch if necessary, e.g., truncate or pad, or raise error
                 min_len = min(num_samples, len(samples_ch))
                 dump_wav[i, :min_len] = samples_ch[:min_len]
            else:
                dump_wav[i, :] = samples_ch

    except Exception as e:
        print(f"Error reading WAV files: {e}")
        return None, None

    # Normalize the multi-channel audio data (optional, consider if needed)
    max_abs = np.max(np.abs(dump_wav))
    if max_abs > 0:
        dump_wav = dump_wav / max_abs * 0.7 # Scale to avoid clipping, 0.7 is arbitrary

    # Create window function (using scipy.signal.windows for clarity)
    # Use frame_len for the window, not fftl unless they are the same
    window = sg.windows.hann(frame_len, sym=False) # Use sym=False for periodic Hann

    # Calculate number of frames
    if num_samples < frame_len:
        print(f"Warning: Number of samples ({num_samples}) is less than frame length ({frame_len}). Cannot compute STFT.")
        return None, num_samples
    number_of_frames = 1 + int((num_samples - frame_len) / shift_len)
    num_freq_bins = int(fftl / 2) + 1

    # Initialize spectrum array
    spectrums = np.zeros((len(channel_vec), number_of_frames, num_freq_bins), dtype=np.complex64)

    # Compute STFT frame by frame
    for i in range(number_of_frames):
        st = i * shift_len
        ed = st + frame_len
        # Ensure frame extraction doesn't go out of bounds
        frame_data = dump_wav[:, st:ed]

        # Apply window (element-wise multiplication)
        # Ensure window length matches frame length
        if frame_data.shape[1] == len(window):
             windowed_frame = frame_data * window
        else:
             # Handle potential mismatch if frame_data is shorter than window (last frame)
             windowed_frame = frame_data * window[:frame_data.shape[1]]


        # Compute FFT
        multi_signal_spectrum = fft(windowed_frame, n=fftl, axis=1)[:, 0:num_freq_bins]
        spectrums[:, i, :] = multi_signal_spectrum

    return spectrums, num_samples

def get_3dim_spectrum_from_data(
    wav_data: np.ndarray,
    frame_len: int,
    shift_len: int,
    fftl: int
) -> Tuple[Optional[np.ndarray], int]:
    """
    Computes the STFT for multi-channel audio data provided as a NumPy array.

    Args:
        wav_data (np.ndarray): Input multi-channel audio data [num_samples x num_channels].
        frame_len (int): Length of each STFT frame (window length).
        shift_len (int): Hop size (shift) between consecutive STFT frames.
        fftl (int): FFT length.

    Returns:
        Tuple[Optional[np.ndarray], int]:
            - Spectrums (np.ndarray | None): 3D array [channel x frame x bin] containing the complex STFT spectrums, or None if input is invalid.
            - num_samples (int): Number of samples in the input data.
    """
    if wav_data.ndim != 2:
        print("Error: Input wav_data must be a 2D array [num_samples x num_channels]")
        return None, 0

    num_samples, num_channels = wav_data.shape
    if num_samples == 0 or num_channels == 0:
        print("Error: Input wav_data has zero samples or channels.")
        return None, 0

    # Transpose to [num_channels x num_samples] for processing consistency
    dump_wav = wav_data.T

    # Normalize (optional)
    max_abs = np.max(np.abs(dump_wav))
    if max_abs > 0:
        dump_wav = dump_wav / max_abs * 0.7

    # Create window function
    window = sg.windows.hann(frame_len, sym=False) # Use sym=False for periodic Hann

    # Calculate number of frames
    if num_samples < frame_len:
         print(f"Warning: Number of samples ({num_samples}) is less than frame length ({frame_len}). Cannot compute STFT.")
         return None, num_samples
    number_of_frames = 1 + int((num_samples - frame_len) / shift_len)
    num_freq_bins = int(fftl / 2) + 1

    # Initialize spectrum array
    spectrums = np.zeros((num_channels, number_of_frames, num_freq_bins), dtype=np.complex64)

    # Compute STFT frame by frame
    for i in range(number_of_frames):
        st = i * shift_len
        ed = st + frame_len
        # Ensure frame extraction doesn't go out of bounds
        frame_data = dump_wav[:, st:ed]

        # Apply window
        # Ensure window length matches frame length
        if frame_data.shape[1] == len(window):
             windowed_frame = frame_data * window
        else:
             # Handle potential mismatch if frame_data is shorter than window (last frame)
             windowed_frame = frame_data * window[:frame_data.shape[1]]

        # Compute FFT
        multi_signal_spectrum = fft(windowed_frame, n=fftl, axis=1)[:, 0:num_freq_bins]
        spectrums[:, i, :] = multi_signal_spectrum

    return spectrums, num_samples


def spec2wav(
    spectrogram: np.ndarray,
    sampling_frequency: int, # Added sampling frequency for potential future use (e.g., saving)
    fftl: int,
    frame_len: int, # Use frame_len (window length) for windowing
    shift_len: int
) -> np.ndarray:
    """
    Converts a single-channel complex spectrogram back to a time-domain waveform using inverse STFT (overlap-add).

    Args:
        spectrogram (np.ndarray): Complex spectrogram [number_of_frames x number_of_bins].
        sampling_frequency (int): Sampling frequency of the original signal.
        fftl (int): FFT length used to create the spectrogram.
        frame_len (int): The original frame/window length used for STFT.
        shift_len (int): The hop size used for STFT.

    Returns:
        np.ndarray: Reconstructed time-domain waveform [num_samples].
    """
    number_of_frames, fft_size = np.shape(spectrogram)
    if fft_size != int(fftl / 2) + 1:
        raise ValueError("Number of bins in spectrogram does not match fftl.")

    # Allocate space for the reconstructed signal
    # Estimated length: (number_of_frames - 1) * shift_len + frame_len
    estimated_len = (number_of_frames - 1) * shift_len + frame_len
    result = np.zeros(estimated_len, dtype=np.float32) # Use float32 for audio

    # Create the analysis window (same as used in STFT)
    # Use frame_len for the window
    hanning_window = sg.windows.hann(frame_len, sym=False)
    # Create synthesis window (needed for perfect reconstruction with Hann window overlap-add)
    # For Hann window with 50% overlap (shift_len = frame_len / 2), a Hann synthesis window works.
    # More generally, a synthesis window needs to satisfy COLA constraint.
    # Assuming 50% overlap for simplicity here. If overlap is different, window needs adjustment.
    if shift_len == frame_len // 2:
        synthesis_window = hanning_window
    else:
        # For non-50% overlap, a simple rectangular window might be used,
        # or a more complex synthesis window calculation is needed for perfect reconstruction.
        # Using Hann window as a default assumption here, may introduce artifacts if overlap isn't 50%.
        print("Warning: Using Hann synthesis window assuming 50% overlap. Reconstruction might not be perfect.")
        synthesis_window = hanning_window


    # Reconstruct full spectrum for each frame and perform inverse FFT
    full_spec_frame = np.zeros(fftl, dtype=np.complex64)
    for i in range(number_of_frames):
        half_spec = spectrogram[i, :]
        # Construct full spectrum using conjugate symmetry
        full_spec_frame[0:fft_size] = half_spec
        # Calculate the indices for the conjugate part carefully
        # Exclude DC (index 0) and Nyquist (index fftl/2 if fftl is even) from flipping
        if fftl % 2 == 0: # Even fftl
             full_spec_frame[fft_size:] = np.conj(half_spec[1:-1][::-1])
        else: # Odd fftl
             full_spec_frame[fft_size:] = np.conj(half_spec[1:][::-1])

        # Inverse FFT
        ifft_frame = np.real(ifft(full_spec_frame, n=fftl))

        # Apply synthesis window
        # Ensure window length matches the iFFT frame length up to frame_len
        windowed_ifft_frame = ifft_frame[:frame_len] * synthesis_window

        # Overlap-add
        start_point = i * shift_len
        end_point = start_point + frame_len
        # Ensure slice indices are within bounds of the result array
        if end_point <= len(result):
            result[start_point:end_point] += windowed_ifft_frame
        else:
            # Handle the last frame potentially exceeding the estimated length
            len_to_add = len(result) - start_point
            if len_to_add > 0:
                 result[start_point:] += windowed_ifft_frame[:len_to_add]


    # Normalize the output signal (optional, depends on windowing and overlap)
    # For Hann window with 50% overlap, the sum of squared windows is constant,
    # but scaling might still be needed depending on FFT normalization.
    # A simple energy-based normalization or peak normalization could be applied if necessary.

    # Trim potential silence at the end if estimated length was too long
    # A more robust way might involve tracking the true signal length if known.
    # For now, returning the calculated length based on frames, shift, and frame length.
    final_len = (number_of_frames - 1) * shift_len + frame_len
    return result[:final_len]



def calculatePos3d(D: float, M: int, theta_degrees: float = 0, r: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to calculate 3D microphone and source positions for a ULA.

    Args:
        D (float): Microphone spacing.
        M (int): Number of microphones.
        theta_degrees (float): Angle in degrees for the desired source direction
                               (relative to broadside, y-axis). Defaults to 0 (broadside).
        r (float): Assumed distance to the source. Defaults to 10.0.

    Returns:
        tuple: (source_position_s, mic_positions_m)
               - source_position_s (np.ndarray): Source position vector [3x1].
               - mic_positions_m (np.ndarray): Microphone positions array [3xM].
    """
    print("(ULA assumption).")
    # Uniform Linear Array (ULA) along the x-axis, centered at origin
    mic_positions_m = np.zeros((3, M))
    mic_positions_m[0, :] = np.linspace(-(M - 1) * D / 2, (M - 1) * D / 2, M)

    # Source position 's' represents the desired look direction.
    # Convert angle (relative to broadside/y-axis) to Cartesian coordinates
    # Angle 0 = broadside (positive y-axis)
    # Angle 90 = endfire (positive x-axis)
    # Angle -90 = endfire (negative x-axis)
    angle_rad = np.radians(90 - theta_degrees) # Convert to standard polar angle (from positive x-axis)

    source_position_s = np.array([
        [r * np.cos(angle_rad)], # x component
        [r * np.sin(angle_rad)], # y component
        [0.0]                    # z component (assuming source in XY plane)
    ])

    return source_position_s, mic_positions_m

