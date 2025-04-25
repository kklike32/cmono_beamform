import torch
import numpy as np
from typing import Tuple
from scipy.io.wavfile import write

# Keep make_mono_audio as NumPy based
import audio_generation
from audio_generation import make_mono_audio
# Keep ULA delay calculation as NumPy based
from audio_generation import create_delay_vector

def set_steering_vector_torch(
    delay_vector: np.ndarray,
    signal_length: int,
    fs: int,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Generates a frequency-domain steering vector using PyTorch.

    Args:
        delay_vector (np.ndarray): Delays (in seconds) for each microphone [num_mics].
        signal_length (int): Length of the time-domain signal (for RFFT frequency calculation).
        fs (int): Sampling frequency in Hz.
        device (torch.device): The device (CPU or CUDA) to create the tensor on.

    Returns:
        torch.Tensor: Complex steering vector [num_freq_bins x num_mics].
    """
    # Calculate frequencies using PyTorch rfftfreq
    freqs = torch.fft.rfftfreq(signal_length, d=1.0/fs, device=device) # Shape [num_freq_bins]
    delay_tensor = torch.from_numpy(delay_vector).to(device).float() # Shape [num_mics]

    # Calculate phase shifts: exp(-1j * 2 * pi * f * delay)
    # Use broadcasting: freqs[:, None] creates [num_freq_bins x 1]
    # delay_tensor is [num_mics], result is [num_freq_bins x num_mics]
    # Ensure complex calculation
    j = torch.complex(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    steering_vector = torch.exp(-j * 2 * torch.pi * freqs.unsqueeze(1) * delay_tensor)
    return steering_vector # Shape [num_freq_bins, num_mics]

def delay_across_channels_torch(
    mono_audio: np.ndarray,
    steering_angle_deg: float,
    num_mics: int,
    mic_separation: float,
    fs: int,
    speed_of_sound: float = 343.0,
    device: torch.device = torch.device('cpu')
) -> np.ndarray:
    """
    Simulates a multi-channel audio signal using PyTorch FFTs.

    Args:
        mono_audio (np.ndarray): Input mono audio signal [num_samples].
        steering_angle_deg (float): Desired steering angle in degrees (0 is broadside).
                                    Positive angle = Right (+x axis).
        num_mics (int): Number of microphones in the array.
        mic_separation (float): Distance between adjacent microphones in meters.
        fs (int): Sampling frequency in Hz.
        speed_of_sound (float): Speed of sound in m/s (default: 343.0 m/s).
        device (torch.device): The device (CPU or CUDA) for computations.

    Returns:
        np.ndarray: Delayed multi-channel audio signal [num_samples x num_mics].
    """
    if mono_audio.ndim != 1:
         if mono_audio.ndim == 2 and mono_audio.shape[1] == 1:
             mono_audio = mono_audio.flatten()
         else:
             raise ValueError("Input mono_audio must be a 1D array.")
    if num_mics <= 0:
        raise ValueError("Number of microphones must be greater than 0!")

    signal_length = len(mono_audio)
    mono_tensor = torch.from_numpy(mono_audio).to(device).float() # Shape [num_samples]

    # Correct angle convention (as per original code's simulation logic)
    steering_angle_deg_corrected = -steering_angle_deg
    angle_rad = np.radians(steering_angle_deg_corrected)

    # Compute delays (NumPy is fine here)
    delay_vec = create_delay_vector(speed_of_sound, angle_rad, num_mics, mic_separation)

    # FFT of the mono audio
    # RFFT input shape: [*, signal_length], output: [*, n_freq_bins]
    mono_audio_f = torch.fft.rfft(mono_tensor) # Shape [num_freq_bins]

    # Generate the frequency-domain steering vector
    steering_vector = set_steering_vector_torch(delay_vec, signal_length, fs, device) # Shape [num_freq_bins, num_mics]

    # Apply phase shifts in the frequency domain
    # mono_audio_f needs unsqueezing: [num_freq_bins, 1]
    # steering_vector: [num_freq_bins, num_mics]
    multi_channel_audio_f = mono_audio_f.unsqueeze(1) * steering_vector # Shape [num_freq_bins, num_mics]

    # Inverse FFT
    # iRFFT input: [*, n_freq_bins], output: [*, signal_length]
    result_tensor = torch.fft.irfft(multi_channel_audio_f, n=signal_length, axis=0) # Specify n and axis

    # Move result back to CPU and convert to NumPy
    return result_tensor.cpu().numpy() # Shape [signal_length, num_mics]
