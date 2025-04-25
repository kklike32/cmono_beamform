import numpy as np
from scipy.io.wavfile import write
from typing import Tuple

"""
Assuming a Uniform Linear Array (ULA) positioned along the x-axis, centered at the origin:

0 degrees: Broadside. The sound source is directly in front of the array, along the positive y-axis. All microphones receive the signal simultaneously (zero delay relative to each other).
Positive Angles (e.g., +30 degrees): The sound source is to the Right of broadside, towards the positive x-axis. Microphones with positive x-coordinates will receive the signal earlier than microphones with negative x-coordinates.
Negative Angles (e.g., -30 degrees): The sound source is to the Left of broadside, towards the negative x-axis. Microphones with negative x-coordinates will receive the signal earlier than microphones with positive x-coordinates.
+90 degrees: Endfire Right. Source is along the positive x-axis.
-90 degrees: Endfire Left. Source is along the negative x-axis.
"""

# ==============================================================================
# Audio Signal Generation
# ==============================================================================

def make_mono_audio(
    frequency: int,
    duration: float = 1.0,
    sampling_rate: int = 16000,
    write_file: bool = False,
    filename: str = "generated_sine.wav"
) -> np.ndarray:
    """
    Generates a mono sine wave audio signal.

    Args:
        frequency (int): Frequency of the sine wave in Hz.
        duration (float): Duration of the generated sine wave in seconds (default: 1.0s).
        sampling_rate (int): Sampling rate in Hz (default: 16000 Hz).
        write_file (bool): If True, saves the generated audio as a WAV file (default: False).
        filename (str): Name of the file to save if write_file is True.

    Returns:
        np.ndarray: NumPy array containing the generated sine wave (float32).
    """
    num_samples = int(sampling_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    result = np.sin(2 * np.pi * frequency * t).astype(np.float32) # Ensure float32 for audio

    if write_file:
        try:
            # Ensure data is scaled appropriately for WAV format if needed,
            # but scipy.io.wavfile handles float32 directly.
            write(filename, sampling_rate, result)
            print(f"Audio saved to {filename}")
        except Exception as e:
            print(f"Error writing WAV file {filename}: {e}")

    return result

def make_noisy_mono_audio(
    frequency: int,
    duration: float = 1.0,
    sampling_rate: int = 16000,
    snr_db: float = 20,
    write_file: bool = False,
    filename: str = "generated_noisy_sine.wav"
) -> np.ndarray:
    """
    Generates a mono sine wave audio signal with additive Gaussian noise.

    Args:
        frequency (int): Frequency of the sine wave in Hz.
        duration (float): Duration of the generated sine wave in seconds (default: 1.0s).
        sampling_rate (int): Sampling rate in Hz (default: 16000 Hz).
        snr_db (float): Signal-to-Noise Ratio in decibels (default: 20 dB).
        write_file (bool): If True, saves the generated audio as a WAV file (default: False).
        filename (str): Name of the file to save if write_file is True.

    Returns:
        np.ndarray: NumPy array containing the generated noisy sine wave (float32).
    """
    num_samples = int(sampling_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Generate noise
    noise = np.random.normal(0, 1, num_samples).astype(np.float32)
    
    # Calculate scaling factor for desired SNR
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    scale = np.sqrt(signal_power / (noise_power * 10**(snr_db/10)))
    
    # Add scaled noise to signal
    result = signal + scale * noise
    
    if write_file:
        try:
            write(filename, sampling_rate, result)
            print(f"Noisy audio saved to {filename}")
        except Exception as e:
            print(f"Error writing WAV file {filename}: {e}")
    
    return result

# ==============================================================================
# Multi-channel Audio Simulation (Plane Wave Assumption)
# ==============================================================================

def create_delay_vector(
    speed_of_sound: float,
    angle_rad: float,
    num_mics: int,
    mic_separation: float
) -> np.ndarray:
    """
    Creates a delay vector for a Uniform Linear Array (ULA) based on steering angle.

    Assumes a plane wave arriving from the specified angle.
    Assumes the ULA is aligned along the x-axis.

    Args:
        speed_of_sound (float): Speed of sound in m/s.
        angle_rad (float): Steering angle in radians (0 is broadside).
        num_mics (int): Number of microphones in the array.
        mic_separation (float): Distance between adjacent microphones in meters.

    Returns:
        np.ndarray: Delay vector (in seconds) for each microphone [num_mics].
                    Positive delay means the signal arrives later.
    """
    # Calculate microphone positions along the x-axis, centered at origin
    mic_indices = np.arange(num_mics)
    mic_positions_x = (mic_indices - (num_mics - 1) / 2.0) * mic_separation

    # Calculate time delay relative to the center of the array
    # Delay = - (mic_pos_x * sin(angle_rad)) / speed_of_sound
    # Note: The sign depends on the angle definition. Here, angle_rad=0 is broadside (y-axis).
    # A positive angle means source is towards positive x-axis.
    # Signal arrives EARLIER at mics with positive x, hence NEGATIVE delay.
    # The original code had -mic_positions * np.sin(angle_rad) / speed_of_sound
    # Let's stick to that convention for consistency with the notebook's likely expectation.
    # Check if mic_positions calculation needs adjustment based on array center.
    # Original notebook used np.arange(num_mics) * mic_separation, assuming mic 0 is at origin.
    # Let's use that for direct compatibility.
    mic_positions = np.arange(num_mics) * mic_separation
    delay_vector = -mic_positions * np.sin(angle_rad) / speed_of_sound
    return delay_vector

def set_steering_vector(
    delay_vector: np.ndarray,
    signal_length: int,
    fs: int
) -> np.ndarray:
    """
    Generates a frequency-domain steering vector for phase shifting based on delays.

    Args:
        delay_vector (np.ndarray): Delays (in seconds) for each microphone [num_mics].
        signal_length (int): Length of the time-domain signal (for RFFT frequency calculation).
        fs (int): Sampling frequency in Hz.

    Returns:
        np.ndarray: Complex steering vector [num_freq_bins x num_mics].
    """
    # Calculate the frequencies for the real FFT (RFFT)
    freqs = np.fft.rfftfreq(signal_length, d=1.0/fs)

    # Calculate phase shifts: exp(-1j * 2 * pi * f * delay)
    # Use broadcasting: freqs[:, np.newaxis] creates [num_freq_bins x 1]
    # delay_vector is [num_mics], result is [num_freq_bins x num_mics]
    steering_vector = np.exp(-1j * 2 * np.pi * freqs[:, np.newaxis] * delay_vector)
    return steering_vector

def delay_across_channels_py_freq(
    mono_audio: np.ndarray,
    steering_angle_deg: float,
    num_mics: int,
    mic_separation: float,
    fs: int,
    speed_of_sound: float = 343.0
) -> np.ndarray:
    """
    Simulates a multi-channel audio signal by delaying a mono signal across microphones.

    Assumes a Uniform Linear Array (ULA) and plane wave propagation.
    Applies delays in the frequency domain via phase shifts.

    Args:
        mono_audio (np.ndarray): Input mono audio signal [num_samples].
        steering_angle_deg (float): Desired steering angle in degrees (0 is broadside).
        num_mics (int): Number of microphones in the array.
        mic_separation (float): Distance between adjacent microphones in meters.
        fs (int): Sampling frequency in Hz.
        speed_of_sound (float): Speed of sound in m/s (default: 343.0 m/s).

    Returns:
        np.ndarray: Delayed multi-channel audio signal [num_samples x num_mics].
    """
    if mono_audio.ndim != 1:
         # If input is already [samples x 1], flatten it. If > 1 channel, raise error.
         if mono_audio.ndim == 2 and mono_audio.shape[1] == 1:
             mono_audio = mono_audio.flatten()
         else:
             raise ValueError("Input mono_audio must be a 1D array.")

    if num_mics <= 0:
        raise ValueError("Number of microphones must be greater than 0!")

    signal_length = len(mono_audio)

    # --- Angle Convention Correction ---
    # Negate the input angle to match the convention used in calculatePos3d
    # where positive angles correspond to the +x direction (Right).
    # The delay calculation tau ~ -x*sin(theta) combined with the phase shift
    # exp(-j*omega*tau) naturally simulates arrival from the *negative* x
    # direction for a *positive* theta. Negating theta flips this.
    steering_angle_deg_corrected = -steering_angle_deg # <--- THIS LINE IS CHANGED/ADDED

    # Convert angle to radians
    angle_rad = np.radians(steering_angle_deg_corrected)

    # Compute delays for each microphone
    delay = create_delay_vector(speed_of_sound, angle_rad, num_mics, mic_separation)

    # FFT of the mono audio (ensure it's treated as a column vector for FFT along axis 0)
    mono_audio_f = np.fft.rfft(mono_audio[:, np.newaxis], axis=0)

    # Generate the frequency-domain steering vector
    steering_vector = set_steering_vector(delay, signal_length, fs) # Shape [num_freq_bins x num_mics]

    # Apply phase shifts in the frequency domain
    # mono_audio_f has shape [num_freq_bins x 1]
    # steering_vector has shape [num_freq_bins x num_mics]
    # Element-wise multiplication works correctly due to broadcasting rules.
    multi_channel_audio_f = mono_audio_f * steering_vector

    # Inverse FFT to get the time-domain multi-channel signal
    # irfft expects shape [..., num_freq_bins], output shape [..., signal_length]
    result = np.fft.irfft(multi_channel_audio_f, n=signal_length, axis=0) # Specify n for exact length

    return result # Shape [signal_length x num_mics]
