import torch
import torchaudio
import numpy as np
import math
from typing import Tuple, Optional

# Constants
SOUND_SPEED = 343.0  # meters per second

def calculate_steering_vector(
    mic_positions: np.ndarray,
    target_doa: Tuple[float, float],
    sample_rate: int,
    n_fft: int,
    sound_speed: float = SOUND_SPEED,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Calculates the steering vector for a given array geometry and DOA.

    Args:
        mic_positions (np.ndarray): Microphone coordinates (num_mics, 3) in meters.
        target_doa (Tuple[float, float]): Target Direction of Arrival (azimuth_deg, elevation_deg).
        sample_rate (int): Audio sample rate in Hz.
        n_fft (int): FFT size used for STFT.
        sound_speed (float): Speed of sound in m/s.
        device (torch.device): PyTorch device.
        dtype (torch.dtype): Complex data type for steering vector.

    Returns:
        torch.Tensor: Steering vector tensor of shape (n_freq_bins, num_mics).
                       n_freq_bins = n_fft // 2 + 1.
    """
    num_mics = mic_positions.shape[0]
    azimuth_rad = np.radians(target_doa[0])
    elevation_rad = np.radians(target_doa[1])

    # Convert DOA to Cartesian unit vector (pointing from origin towards source)
    # x: front, y: left, z: up
    doa_vector = np.array([
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.sin(elevation_rad)
    ])

    # Calculate time delays (tau) for each microphone relative to the origin
    # tau = (mic_pos . doa_vector) / sound_speed
    # Note the sign: positive delay for mics further away in the DOA direction
    # We use -doa_vector for wave propagation direction relative to mic positions
    taus = -np.dot(mic_positions, doa_vector) / sound_speed
    taus = torch.from_numpy(taus).to(device=device, dtype=torch.float32) # Shape: (num_mics,)

    # Calculate frequency bins
    freq_bins = torch.fft.rfftfreq(n_fft, d=1.0/sample_rate).to(device=device, dtype=torch.float32) # Shape: (n_freq_bins,)
    n_freq_bins = freq_bins.shape[0]

    # Calculate angular frequencies (omega = 2 * pi * f)
    omega = 2 * math.pi * freq_bins # Shape: (n_freq_bins,)

    # Calculate steering vector: exp(j * omega * tau)
    # We need shapes: (n_freq_bins, 1) and (1, num_mics) for broadcasting
    steering_vector = torch.exp(1j * omega.unsqueeze(1) * taus.unsqueeze(0)) # Shape: (n_freq_bins, num_mics)

    return steering_vector.to(dtype=dtype)


def compute_scm(
    stft_signal: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the Spatial Covariance Matrix (SCM) for each frequency bin.

    Args:
        stft_signal (torch.Tensor): Complex STFT signal (num_channels, n_freq_bins, n_frames).
        mask (Optional[torch.Tensor]): Time-frequency mask (n_freq_bins, n_frames)
                                        to select frames/bins for SCM calculation (e.g., noise mask).
                                        If None, uses all frames.

    Returns:
        torch.Tensor: Spatial Covariance Matrix (SCM) tensor (n_freq_bins, num_channels, num_channels).
    """
    n_channels, n_freq_bins, n_frames = stft_signal.shape

    # Permute to (n_freq_bins, n_frames, n_channels) for easier SCM calculation per bin
    stft_permuted = stft_signal.permute(1, 2, 0) # Shape: (n_freq_bins, n_frames, n_channels)

    if mask is not None:
        # Ensure mask is broadcastable: (n_freq_bins, n_frames, 1)
        mask = mask.unsqueeze(-1)
        # Apply mask and count effective frames per frequency bin
        stft_permuted = stft_permuted * mask
        effective_frames = torch.sum(mask, dim=1, keepdim=True).clamp(min=1e-8) # Shape: (n_freq_bins, 1, 1)
    else:
        effective_frames = n_frames

    # Calculate SCM: R = E[X X^H] = (1/N) * sum(X_t X_t^H)
    # X is (n_freq_bins, n_frames, n_channels)
    # X^H (conjugate transpose) is (n_freq_bins, n_channels, n_frames)
    # We need batch matrix multiplication for each frequency bin
    # einsum is efficient: '...tc,...kc->...tk' where t=time, c=channel1, k=channel2
    scm = torch.einsum('...tc,...kc->...ck', stft_permuted, stft_permuted.conj()) / effective_frames
    # Result shape: (n_freq_bins, n_channels, n_channels)

    return scm


def perform_pytorch_mvdr(
    audio_data: np.ndarray,
    sample_rate: int,
    mic_positions: np.ndarray,
    target_doa: Tuple[float, float],
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    reference_channel: int = 0,
    diagonal_loading_param: float = 1e-6,
    sound_speed: float = SOUND_SPEED,
    use_diagonal_noise_scm: bool = True,
    noise_scm_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Performs MVDR beamforming using PyTorch/torchaudio.

    Args:
        audio_data (np.ndarray): Multi-channel audio data (num_samples, num_channels).
        sample_rate (int): Audio sample rate in Hz.
        mic_positions (np.ndarray): Microphone coordinates (num_mics, 3) in meters.
        target_doa (Tuple[float, float]): Target Direction of Arrival (azimuth_deg, elevation_deg).
        n_fft (int): FFT size for STFT.
        hop_length (Optional[int]): Hop length for STFT. Defaults to n_fft // 4.
        reference_channel (int): Index of the reference microphone.
        diagonal_loading_param (float): Small value added to the diagonal of the noise SCM
                                         for numerical stability.
        sound_speed (float): Speed of sound in m/s.
        use_diagonal_noise_scm (bool): If True, estimates a diagonal noise SCM based on
                                       average power per frequency bin. If False, uses the
                                       full SCM computed from the input (less ideal for MVDR
                                       unless a proper noise mask is provided).
        noise_scm_mask (Optional[np.ndarray]): Time-frequency mask (bool or float, shape
                                               n_freq_bins x n_frames) indicating noise-only
                                               regions for SCM estimation. If provided,
                                               `use_diagonal_noise_scm` is ignored and the
                                               full SCM is computed using the mask.

    Returns:
        np.ndarray: Single-channel beamformed audio data (num_samples,).
    """
    if hop_length is None:
        hop_length = n_fft // 4

    num_samples, num_channels = audio_data.shape
    if num_channels != mic_positions.shape[0]:
        raise ValueError("Number of channels in audio_data must match number of microphones in mic_positions.")

    # --- Device and Data Type ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    complex_dtype = torch.complex64
    float_dtype = torch.float32

    # --- Convert to Tensor ---
    audio_tensor = torch.from_numpy(audio_data).to(device=device, dtype=float_dtype).T # Shape: (num_channels, num_samples)

    # --- STFT ---
    window = torch.hann_window(n_fft, device=device, dtype=float_dtype)
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        window_fn=lambda x: torch.hann_window(x, periodic=True, device=device, dtype=float_dtype), # Use periodic Hann for overlap-add
        power=None, # Return complex result
        center=True,
        pad_mode="constant", # or 'reflect'
        normalized=False, # Standard STFT definition
    ).to(device)

    stft_signal = stft_transform(audio_tensor) # Shape: (num_channels, n_freq_bins, n_frames)
    n_freq_bins = stft_signal.shape[1]
    n_frames = stft_signal.shape[2]
    print(f"STFT shape: {stft_signal.shape}")

    # --- Calculate Steering Vector ---
    steering_vector = calculate_steering_vector(
        mic_positions, target_doa, sample_rate, n_fft, sound_speed, device, complex_dtype
    ) # Shape: (n_freq_bins, num_mics)
    print(f"Steering vector shape: {steering_vector.shape}")

    # --- Estimate Noise Spatial Covariance Matrix (SCM) ---
    noise_scm_mask_tensor = None
    if noise_scm_mask is not None:
        if noise_scm_mask.shape != (n_freq_bins, n_frames):
             raise ValueError(f"noise_scm_mask shape mismatch. Expected {(n_freq_bins, n_frames)}, got {noise_scm_mask.shape}")
        print("Using provided mask to compute full noise SCM.")
        noise_scm_mask_tensor = torch.from_numpy(noise_scm_mask).to(device=device, dtype=float_dtype)
        # Compute full SCM using only masked regions
        noise_scm = compute_scm(stft_signal, mask=noise_scm_mask_tensor)
    elif use_diagonal_noise_scm:
        print("Estimating diagonal noise SCM from mixture average power.")
        # Estimate SCM from the *entire* mixture first
        mixture_scm = compute_scm(stft_signal) # Shape: (n_freq_bins, num_channels, num_channels)
        # Estimate noise power per frequency bin as the average power across channels
        # Use diagonal elements of mixture SCM (power at each mic) and average them
        noise_power_per_freq = torch.mean(torch.diagonal(mixture_scm, dim1=-2, dim2=-1).real, dim=-1) # Shape: (n_freq_bins,)
        # Create diagonal noise SCM: Rnn = diag(noise_power_per_freq) * I
        # Ensure positive definite by clamping small values
        noise_power_per_freq = torch.clamp(noise_power_per_freq, min=1e-8)
        noise_scm = torch.diag_embed(
            torch.ones(num_channels, device=device, dtype=complex_dtype)
        ).unsqueeze(0) * noise_power_per_freq.view(-1, 1, 1) # Shape: (n_freq_bins, num_channels, num_channels)
    else:
        print("WARNING: Using full mixture SCM as noise SCM proxy. This might suppress the target.")
        # Use the SCM computed from the entire signal as a proxy for noise SCM
        # This is often suboptimal for MVDR as it includes the target signal.
        noise_scm = compute_scm(stft_signal) # Shape: (n_freq_bins, num_channels, num_channels)

    # Apply diagonal loading for numerical stability (regularization)
    # Add eps * I to the diagonal
    identity = torch.eye(num_channels, device=device, dtype=complex_dtype).unsqueeze(0) # Shape: (1, num_channels, num_channels)
    noise_scm = noise_scm + diagonal_loading_param * identity
    print(f"Noise SCM shape: {noise_scm.shape}")

    # --- Calculate MVDR Weights ---
    # torchaudio expects Relative Transfer Function (RTF) which is related to steering vector.
    # For MVDR, RTF can often be directly substituted by the steering vector.
    # mvdr_weights_rtf(rtf, psd_noise, reference_channel)
    try:
        mvdr_weights = torchaudio.functional.mvdr_weights_rtf(
            rtf=steering_vector,
            psd_noise=noise_scm,
            reference_channel=reference_channel,
            # diagonal_loading=True, # Already applied manually
            # diag_eps=diagonal_loading_param # Already applied manually
        ) # Shape: (n_freq_bins, num_channels)
        print(f"MVDR weights shape: {mvdr_weights.shape}")
    except Exception as e:
        print(f"Error calculating MVDR weights: {e}")
        print("Common issues: SCM not positive definite (try increasing diagonal_loading_param), shapes mismatch.")
        # As a fallback, return zeros or raise error
        return np.zeros(num_samples)


    # --- Apply Beamforming Weights ---
    # Formula: Y(f, t) = sum_c(W(f, c)^* * X(c, f, t))
    # Need to match shapes for broadcasting or use einsum.
    # W shape: (n_freq_bins, num_channels) -> W.conj()
    # X shape: (num_channels, n_freq_bins, n_frames)
    # Target Y shape: (n_freq_bins, n_frames)

    # Using torch.einsum for clarity and efficiency:
    # 'fc,cft->ft' where f=freq, c=channel, t=time
    stft_beamformed = torch.einsum('fc,cft->ft', mvdr_weights.conj(), stft_signal)
    # Shape: (n_freq_bins, n_frames)
    print(f"Beamformed STFT shape: {stft_beamformed.shape}")


    # --- Inverse STFT ---
    # Need to adjust length for istft consistency if center=True was used in stft
    output_length = num_samples

    istft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        window_fn=lambda x: torch.hann_window(x, periodic=True, device=device, dtype=float_dtype), # Must match STFT window
        normalized=False, # Match STFT
    ).to(device)

    beamformed_audio = istft_transform(stft_beamformed, length=output_length) # Shape: (num_samples,)
    print(f"Output audio shape: {beamformed_audio.shape}")

    # --- Convert back to NumPy ---
    beamformed_audio_np = beamformed_audio.cpu().numpy()

    return beamformed_audio_np


# --- Example Usage (for testing the script directly) ---
if __name__ == "__main__":
    print("Running PyTorch MVDR Analysis Script Self-Test...")

    # --- Dummy Data Configuration ---
    SAMPLE_RATE = 16000
    DURATION_S = 5
    N_SAMPLES = SAMPLE_RATE * DURATION_S
    N_CHANNELS = 4
    N_FFT = 1024
    HOP_LENGTH = N_FFT // 4

    # Simple linear array on x-axis
    MIC_POSITIONS = np.array([
        [-0.05, 0, 0],
        [-0.015, 0, 0],
        [0.015, 0, 0],
        [0.05, 0, 0]
    ])

    # Target DOA (e.g., 30 degrees azimuth, 0 degrees elevation)
    TARGET_DOA = (30.0, 0.0)

    # --- Generate Dummy Audio (e.g., noise + delayed sine wave) ---
    print("Generating dummy audio data...")
    times = np.arange(N_SAMPLES) / SAMPLE_RATE
    # Simple sine wave as target signal
    target_signal = 0.5 * np.sin(2 * np.pi * 440 * times)

    # Calculate delays for the target DOA
    az_rad = np.radians(TARGET_DOA[0])
    el_rad = np.radians(TARGET_DOA[1])
    doa_vec = np.array([np.cos(el_rad) * np.cos(az_rad), np.cos(el_rad) * np.sin(az_rad), np.sin(el_rad)])
    taus = -np.dot(MIC_POSITIONS, doa_vec) / SOUND_SPEED
    print(f"Delays for target DOA: {taus * 1000:.2f} ms")

    # Create multi-channel audio by delaying the target signal and adding noise
    multi_channel_audio = np.zeros((N_SAMPLES, N_CHANNELS))
    for i in range(N_CHANNELS):
        delay_samples = int(round(taus[i] * SAMPLE_RATE))
        delayed_signal = np.zeros_like(target_signal)
        if delay_samples >= 0:
            delayed_signal[delay_samples:] = target_signal[:-delay_samples]
        else:
            delayed_signal[:delay_samples] = target_signal[-delay_samples:]
        multi_channel_audio[:, i] = delayed_signal

    # Add some noise
    noise_power = 0.1
    multi_channel_audio += noise_power * np.random.randn(N_SAMPLES, N_CHANNELS)
    print(f"Dummy audio data shape: {multi_channel_audio.shape}")

    # --- Perform PyTorch MVDR ---
    print("\nPerforming PyTorch MVDR...")
    beamformed_output = perform_pytorch_mvdr(
        audio_data=multi_channel_audio,
        sample_rate=SAMPLE_RATE,
        mic_positions=MIC_POSITIONS,
        target_doa=TARGET_DOA,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        reference_channel=0,
        use_diagonal_noise_scm=True # Try True and False
    )

    print(f"\nBeamforming complete. Output shape: {beamformed_output.shape}")

    # Optional: Save or plot results here if running standalone
    # import soundfile as sf
    # sf.write("pytorch_mvdr_output.wav", beamformed_output, SAMPLE_RATE)
    # print("Saved beamformed output to pytorch_mvdr_output.wav")

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(times, multi_channel_audio[:, 0], label='Mic 0 (Ref)')
    # plt.plot(times, beamformed_output, label='PyTorch MVDR Output', alpha=0.7)
    # plt.title("Dummy Audio and PyTorch MVDR Output")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

