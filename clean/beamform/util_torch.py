import torch
import numpy as np
from typing import Tuple, Optional

def get_3dim_spectrum_torch(
    wav_data: np.ndarray,
    frame_len: int,
    shift_len: int,
    fftl: int,
    device: torch.device = torch.device('cpu')
) -> Tuple[Optional[torch.Tensor], int]:
    """
    Computes the STFT for multi-channel audio data using PyTorch.

    Args:
        wav_data (np.ndarray): Input multi-channel audio data [num_samples x num_channels].
        frame_len (int): Length of each STFT frame (window length).
        shift_len (int): Hop size (shift) between consecutive STFT frames.
        fftl (int): FFT length.
        device (torch.device): The device (CPU or CUDA) to perform computations on.

    Returns:
        Tuple[Optional[torch.Tensor], int]:
            - Spectrums (torch.Tensor | None): 3D tensor [channel x freq_bin x frame]
                                               containing the complex STFT spectrums, or None if input is invalid.
            - num_samples (int): Number of samples in the input data.
    """
    if wav_data.ndim != 2:
        print("Error: Input wav_data must be a 2D array [num_samples x num_channels]")
        return None, 0

    num_samples, num_channels = wav_data.shape
    if num_samples == 0 or num_channels == 0:
        print("Error: Input wav_data has zero samples or channels.")
        return None, 0
    if num_samples < frame_len:
         print(f"Warning: Number of samples ({num_samples}) is less than frame length ({frame_len}). Cannot compute STFT.")
         return None, num_samples

    # Convert NumPy array to PyTorch tensor and move to device
    # Transpose to [num_channels x num_samples] for PyTorch STFT
    wav_tensor = torch.from_numpy(wav_data.T).to(device).float() # Ensure float32

    # Create window tensor
    window = torch.hann_window(frame_len, periodic=True, device=device) # Match scipy's sym=False

    # Compute STFT
    # torch.stft expects input shape [*, n_fft] or [batch, n_fft] or [batch, time]
    # For multi-channel, we process each channel independently.
    # It's often easier to pass [batch, time] where batch=channels.
    try:
        stft_result = torch.stft(
            wav_tensor,
            n_fft=fftl,
            hop_length=shift_len,
            win_length=frame_len,
            window=window,
            center=False, # Match numpy/scipy behavior (no padding at ends by default)
            return_complex=True
        )
        # Output shape: [num_channels, num_freq_bins, num_frames]
        return stft_result, num_samples
    except Exception as e:
        print(f"Error during PyTorch STFT: {e}")
        return None, num_samples

def spec2wav_torch(
    spectrogram: torch.Tensor,
    fftl: int,
    frame_len: int,
    shift_len: int,
    num_samples_original: Optional[int] = None, # Optional: for exact length trimming
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Converts a single-channel complex spectrogram tensor back to a time-domain waveform using PyTorch iSTFT.

    Args:
        spectrogram (torch.Tensor): Complex spectrogram tensor [num_freq_bins x num_frames].
                                    Must be on the specified device.
        fftl (int): FFT length used to create the spectrogram.
        frame_len (int): The original frame/window length used for STFT.
        shift_len (int): The hop size used for STFT.
        num_samples_original (Optional[int]): Original number of samples to trim the output.
        device (torch.device): The device (CPU or CUDA) the tensor is on.

    Returns:
        torch.Tensor: Reconstructed time-domain waveform [num_samples].
    """
    if not torch.is_complex(spectrogram):
        raise ValueError("Input spectrogram must be a complex tensor.")
    if spectrogram.dim() != 2:
        raise ValueError("Input spectrogram must be 2D [num_freq_bins x num_frames].")

    # Create window tensor
    window = torch.hann_window(frame_len, periodic=True, device=device)
    
    # Ensure the window has proper normalization for correct overlap-add
    # This is crucial to avoid the "window overlap add min" error
    S = sum(window[i] * window[i + shift_len] for i in range(0, frame_len - shift_len))
    if S < 0.999:  # If window overlap is insufficient
        print(f"Warning: Window overlap sum is {S}, adjusting window")
        # Normalize window for proper overlap-add
        window = window / torch.sqrt(S * 2)
    
    try:
        # Ensure the spectrogram has the right shape for istft
        # PyTorch istft expects [freq_bins, time_frames]
        if spectrogram.shape[0] != fftl // 2 + 1:
            raise ValueError(f"Expected first dimension to be {fftl // 2 + 1}, got {spectrogram.shape[0]}")
        
        # Add a batch dimension for istft
        spectrogram_batch = spectrogram.unsqueeze(0)
        
        waveform = torch.istft(
            spectrogram_batch,
            n_fft=fftl,
            hop_length=shift_len,
            win_length=frame_len,
            window=window,
            center=False,
            length=num_samples_original,
            return_complex=False  # Ensure we return a real tensor
        )
        
        # Return the result directly as a tensor
        return waveform.squeeze(0)
    except Exception as e:
        print(f"Error during PyTorch iSTFT: {e}")
        # Return a zero tensor instead of empty array
        return torch.zeros(num_samples_original if num_samples_original else 1, 
                          device=device, dtype=torch.float32)


def stab(mat: torch.Tensor, theta: float, num_channels: int) -> torch.Tensor:
    """
    Stabilizes a matrix by adding a scaled identity matrix if its condition number is too high.
    PyTorch implementation of the NumPy version in util.py.

    Args:
        mat (torch.Tensor): The complex square matrix to stabilize [num_channels x num_channels].
        theta (float): The condition number threshold. If cond(mat) > theta, stabilization is applied.
        num_channels (int): The number of channels (dimension of the matrix).

    Returns:
        torch.Tensor: The stabilized matrix.
    """
    # Ensure we have a PyTorch tensor
    if not isinstance(mat, torch.Tensor):
        raise TypeError("Input 'mat' must be a PyTorch tensor")
    
    # Get device of input tensor
    device = mat.device
    
    # Create a vector of decreasing powers of 10 for stabilization factor
    d = torch.pow(10.0, torch.arange(-num_channels, 0, dtype=torch.float32, device=device))
    d = d.to(mat.dtype)  # Match input dtype (e.g., complex64)

    result_mat = mat.clone()  # Work on a copy to avoid modifying the original matrix
    identity_mat = torch.eye(num_channels, dtype=mat.dtype, device=device)

    # Calculate initial condition number
    try:
        cond_num = torch.linalg.cond(result_mat)
    except torch.linalg.LinAlgError:
        # Handle cases where the matrix might be singular initially
        cond_num = torch.tensor(float('inf'), device=device)

    # Iteratively add scaled identity matrix until condition number is below threshold
    for i in range(num_channels):
        if cond_num <= theta:
            break  # Matrix is stable enough
        # Add diagonal loading
        result_mat += d[i] * identity_mat
        # Recalculate condition number
        try:
            cond_num = torch.linalg.cond(result_mat)
        except torch.linalg.LinAlgError:
            cond_num = torch.tensor(float('inf'), device=device)  # Matrix became singular during stabilization attempt

    # If the loop finished without stabilizing below theta, provide a warning
    if cond_num > theta:
         print(f"Warning: Matrix stabilization might not have fully succeeded. Final condition number: {cond_num.item()}")

    return result_mat


def get_device():
    """
    Returns the best available device (CUDA GPU if available, otherwise CPU).
    
    Returns:
        torch.device: The best available device for PyTorch operations
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
