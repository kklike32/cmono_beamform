import numpy as np
import torch
from typing import Tuple, Optional
from . import util_torch
# from . import util # util is not used directly here

class MinimumVarianceDistortionlessResponseTorch:
    """
    Implements the MVDR beamformer using PyTorch.
    Assumes microphone positions are provided in 3D space.
    """
    def __init__(self,
                 mic_positions_m: np.ndarray,
                 sampling_frequency: int = 16000,
                 sound_speed: float = 343.0,
                 fft_length: int = 512,
                 fft_shift: int = 256,
                 device: torch.device = None):
        """
        Initializes the PyTorch MVDR beamformer.

        Args:
            mic_positions_m (np.ndarray): Array of microphone positions [3 x num_mics].
            sampling_frequency (int): Sampling frequency in Hz.
            sound_speed (float): Speed of sound in m/s.
            fft_length (int): FFT length for STFT processing.
            fft_shift (int): Hop size (shift) for STFT processing.
            device (torch.device): The device (CPU or CUDA) for computations.
                                  If None, the best available device will be used.
        """
        if mic_positions_m.ndim != 2 or mic_positions_m.shape[0] != 3:
            raise ValueError("mic_positions_m must be a 3xM array (M = number of mics)")
        
        # Auto-detect device if not specified
        if device is None:
            device = util_torch.get_device()
            print(f"Using device: {device}")
        
        self.device = device
        self.mic_positions_m = torch.from_numpy(mic_positions_m).to(device).float() # Store as tensor
        self.num_mics = self.mic_positions_m.shape[1]
        self.sampling_frequency = sampling_frequency
        self.sound_speed = sound_speed
        self.fft_length = fft_length
        self.fft_shift = fft_shift
        self.frame_len = fft_length
        self.num_freq_bins = int(fft_length / 2) + 1
        self.j = torch.complex(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))


    def get_steering_vector_torch(self, source_position_s: np.ndarray) -> torch.Tensor:
        """
        Calculates the steering vector using PyTorch.

        Args:
            source_position_s (np.ndarray): Source position vector [3 x 1] in meters.

        Returns:
            torch.Tensor: Complex steering vector [num_mics x num_freq_bins] on the specified device.
        """
        if source_position_s.shape != (3, 1):
             raise ValueError("source_position_s must be a 3x1 vector")

        source_pos_tensor = torch.from_numpy(source_position_s).to(self.device).float() # Shape [3, 1]

        # Frequencies (only positive needed)
        freq_vector = torch.linspace(0, self.sampling_frequency / 2, self.num_freq_bins, device=self.device) # Shape [num_freq_bins]

        # Calculate distances from source 's' to each microphone 'm_i'
        # mic_positions_m is [3, M]
        # source_pos_tensor is [3, 1]
        distances = torch.sqrt(torch.sum((self.mic_positions_m - source_pos_tensor) ** 2, dim=0)) # Shape [M]

        # Calculate time delays relative to furthest mic (ensure non-negative delays)
        max_dist = torch.max(distances)
        time_delays_positive = (max_dist - distances) / self.sound_speed # Shape [M], delays >= 0

        # Calculate steering vector components: exp(-1j * 2 * pi * f * t_positive)
        # Use broadcasting:
        # time_delays_positive [M] -> [M, 1]
        # freq_vector [num_freq_bins] -> [1, num_freq_bins]
        # phase_term [M, num_freq_bins]
        phase_term = -self.j * 2 * torch.pi * time_delays_positive.unsqueeze(1) * freq_vector.unsqueeze(0)
        steering_vector = torch.exp(phase_term) # Shape [num_mics, num_freq_bins]

        return steering_vector

    def get_spatial_correlation_matrix_torch(
        self,
        multi_signal_tensor: torch.Tensor, # Expects [num_channels, num_samples]
        use_number_of_frames_init: int = 10,
        use_number_of_frames_final: int = 10
        ) -> torch.Tensor:
        """
        Calculates the Spatial Correlation Matrix (SCM) using PyTorch STFT and tensor operations.
        Averages over initial and final frames (intended for noise estimation).

        Args:
            multi_signal_tensor (torch.Tensor): Multi-channel time-domain signal tensor
                                               [num_channels x num_samples] on the specified device.
            use_number_of_frames_init (int): Number of initial frames to use.
            use_number_of_frames_final (int): Number of final frames to use.

        Returns:
            torch.Tensor: The estimated SCM tensor [num_mics x num_mics x num_freq_bins] on device.
        """
        if multi_signal_tensor.shape[0] != self.num_mics:
             raise ValueError("First dimension of multi_signal_tensor must match num_mics.")
        if multi_signal_tensor.device != self.device:
             raise ValueError("Input tensor must be on the same device as the beamformer.")

        # Perform STFT
        window = torch.hann_window(self.frame_len, periodic=True, device=self.device)
        stft_result = torch.stft(
            multi_signal_tensor,
            n_fft=self.fft_length,
            hop_length=self.fft_shift,
            win_length=self.frame_len,
            window=window,
            center=False,
            return_complex=True
        )
        # stft_result shape: [num_channels, num_freq_bins, num_frames]

        num_channels, num_bins, num_frames_total = stft_result.shape

        if num_frames_total == 0:
             print("Warning: STFT resulted in zero frames.")
             return torch.zeros((self.num_mics, self.num_mics, self.num_freq_bins),
                                dtype=torch.complex64, device=self.device)

        # Ensure frame counts are non-negative and within bounds
        use_number_of_frames_init = max(0, min(use_number_of_frames_init, num_frames_total))
        use_number_of_frames_final = max(0, min(use_number_of_frames_final, num_frames_total))

        # Select frames for averaging, handling overlap and edge cases
        if use_number_of_frames_init == 0 and use_number_of_frames_final == 0:
            print("Warning: Both init and final frame counts are zero. Using all frames for SCM.")
            frames_to_avg = stft_result
        elif use_number_of_frames_init + use_number_of_frames_final > num_frames_total:
            # Overlap or counts exceed total frames, use all frames
            print("Warning: SCM init/final frame counts exceed total frames or overlap significantly. Using all frames.")
            frames_to_avg = stft_result
        else:
            # No overlap, concatenate distinct initial and final frames
            frames_init = stft_result[:, :, :use_number_of_frames_init]
            frames_final = stft_result[:, :, (num_frames_total - use_number_of_frames_final):]
            frames_to_avg = torch.cat((frames_init, frames_final), dim=2)


        num_frames_avg = frames_to_avg.shape[2]
        if num_frames_avg == 0:
            print("Warning: No frames selected for SCM averaging.")
            return torch.zeros((self.num_mics, self.num_mics, self.num_freq_bins),
                               dtype=torch.complex64, device=self.device)

        # Calculate SCM: Average outer product -> R = E[X(f)X(f)^H]
        # X has shape [channels, bins, frames]
        # We want R of shape [channels, channels, bins]
        # Use torch.einsum for clarity: 'cft,dft->fcd' sums over frames 't'
        # c, d = channels; f = bins; t = frames
        # Need to permute frames_to_avg to [channels, bins, frames] -> [bins, channels, frames]
        X = frames_to_avg.permute(1, 0, 2) # Shape [bins, channels, frames]
        # Calculate outer product and sum over frames
        # Use conjugate for Hermitian transpose: X.conj()
        scm = torch.einsum('fct,fdt->fcd', X, X.conj()) / num_frames_avg

        # Permute SCM to match expected [channels, channels, bins]
        scm = scm.permute(1, 2, 0) # Shape [channels, channels, bins]

        return scm


    def get_mvdr_beamformer_torch(
        self,
        steering_vector: torch.Tensor, # Shape [num_mics, num_freq_bins]
        scm_matrix: torch.Tensor,      # Shape [num_mics, num_mics, num_freq_bins]
        stabilization_theta: float = 1e5
        ) -> torch.Tensor:
        """
        Calculates the MVDR beamformer weights using PyTorch.

        Args:
            steering_vector (torch.Tensor): Steering vector [num_mics x num_freq_bins].
            scm_matrix (torch.Tensor): Spatial Correlation Matrix [num_mics x num_mics x num_freq_bins].
            stabilization_theta (float): Condition number threshold for SCM stabilization.

        Returns:
            torch.Tensor: MVDR beamformer weights [num_mics x num_freq_bins].
        """
        if steering_vector.shape != (self.num_mics, self.num_freq_bins) or \
           scm_matrix.shape != (self.num_mics, self.num_mics, self.num_freq_bins):
             raise ValueError("Shape mismatch in input tensors.")
        if steering_vector.device != self.device or scm_matrix.device != self.device:
             raise ValueError("Input tensors must be on the same device as the beamformer.")

        beamformer_weights = torch.zeros_like(steering_vector, dtype=torch.complex64)

        for f_idx in range(self.num_freq_bins):
            # Extract SCM and steering vector for this frequency bin
            R_f = scm_matrix[:, :, f_idx] # Shape [num_mics, num_mics]
            d_f = steering_vector[:, f_idx:f_idx+1] # Keep as column vector [num_mics, 1]

            # Use PyTorch stabilization (faster than CPU conversion)
            R_f_stable = util_torch.stab(R_f, stabilization_theta, self.num_mics)

            # Invert the SCM
            try:
                # Use pseudo-inverse for robustness, similar to NumPy's pinv
                inv_R_f = torch.linalg.pinv(R_f_stable)
            except torch.linalg.LinAlgError:
                 print(f"Warning: Matrix inversion failed for frequency bin {f_idx}. Using identity matrix.")
                 inv_R_f = torch.eye(self.num_mics, dtype=R_f_stable.dtype, device=self.device)

            # Calculate numerator: R(f)^-1 @ d(f)
            numerator = torch.matmul(inv_R_f, d_f) # Shape [num_mics, 1]

            # Calculate denominator: d(f)^H @ R(f)^-1 @ d(f)
            # d_f_H is conjugate transpose [1, num_mics]
            d_f_H = d_f.conj().T
            denominator = torch.matmul(d_f_H, numerator) # Shape [1, 1] (scalar complex tensor)

            # Avoid division by zero or very small numbers
            if torch.abs(denominator) > 1e-10:
                 # Apply formula: w = numerator / denominator
                 beamformer_weights[:, f_idx] = numerator.squeeze() / denominator.squeeze()
            else:
                 # Handle case where denominator is near zero
                 print(f"Warning: Denominator near zero for frequency bin {f_idx}. Setting weights to zero.")
                 beamformer_weights[:, f_idx] = 0.0 # Set weights to zero for this bin

        return beamformer_weights # Shape [num_mics, num_freq_bins]
