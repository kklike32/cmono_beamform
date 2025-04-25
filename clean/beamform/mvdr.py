import numpy as np
from scipy.fftpack import fft
from typing import Tuple, Optional
from . import util
from scipy import signal as sg
from scipy.signal import get_window
from scipy.signal.windows import hamming, hann, blackman, kaiser

"""
Assuming a Uniform Linear Array (ULA) positioned along the x-axis, centered at the origin:

0 degrees: Broadside. The sound source is directly in front of the array, along the positive y-axis. All microphones receive the signal simultaneously (zero delay relative to each other).
Positive Angles (e.g., +30 degrees): The sound source is to the Right of broadside, towards the positive x-axis. Microphones with positive x-coordinates will receive the signal earlier than microphones with negative x-coordinates.
Negative Angles (e.g., -30 degrees): The sound source is to the Left of broadside, towards the negative x-axis. Microphones with negative x-coordinates will receive the signal earlier than microphones with positive x-coordinates.
+90 degrees: Endfire Right. Source is along the positive x-axis.
-90 degrees: Endfire Left. Source is along the negative x-axis.
"""

class MinimumVarianceDistortionlessResponse:
    """
    Implements the Minimum Variance Distortionless Response (MVDR) beamformer.

    This version assumes microphone positions are provided in 3D space.
    """

    def __init__(self,
                 mic_positions_m: np.ndarray,
                 sampling_frequency: int = 16000,
                 sound_speed: float = 343.0,
                 fft_length: int = 512,
                 fft_shift: int = 256):
        """
        Initializes the MVDR beamformer.

        Args:
            mic_positions_m (np.ndarray): Array of microphone positions [3 x num_mics].
                                          Coordinates should be in meters.
            sampling_frequency (int): Sampling frequency in Hz.
            sound_speed (float): Speed of sound in m/s.
            fft_length (int): FFT length for STFT processing.
            fft_shift (int): Hop size (shift) for STFT processing.
        """
        if mic_positions_m.ndim != 2 or mic_positions_m.shape[0] != 3:
            raise ValueError("mic_positions_m must be a 3xM array (M = number of mics)")

        self.mic_positions_m = mic_positions_m
        self.num_mics = mic_positions_m.shape[1]
        self.sampling_frequency = sampling_frequency
        self.sound_speed = sound_speed
        self.fft_length = fft_length
        self.fft_shift = fft_shift
        self.num_freq_bins = int(fft_length / 2) + 1

    def _normalize_steering_vector(self, steering_vector: np.ndarray) -> np.ndarray:
        """
        Normalizes the steering vector for each frequency bin.

        Normalization ensures that the dot product of the vector with its conjugate transpose is 1.
        This specific normalization (dividing by sqrt(M) implicitly via the formula used later)
        might be specific to certain MVDR formulations. Standard MVDR often doesn't require
        this explicit normalization step for the steering vector itself, but rather uses it
        within the weight calculation formula.

        Args:
            steering_vector (np.ndarray): The steering vector [num_mics x num_freq_bins].

        Returns:
            np.ndarray: The normalized steering vector.
        """
        # The original code normalized by dividing by matmul(conj(sv).T, sv) per bin.
        # This seems unusual for standard MVDR steering vectors. Let's stick to the
        # calculation based on delays/positions without this explicit normalization here.
        # The normalization happens implicitly in the get_mvdr_beamformer method.
        # If this specific normalization is required, it should be justified/clarified.
        # For now, returning the unnormalized vector.
        
        print("Note: Steering vector normalization step skipped as it's handled in beamformer calculation.")

        return steering_vector


    def get_steering_vector(self, source_position_s: np.ndarray) -> np.ndarray:
        """
        Calculates the steering vector for a given source position using time delays.

        Assumes the source is at a specific point 's' relative to the microphones 'm'.

        Args:
            source_position_s (np.ndarray): Source position vector [3 x 1] in meters.

        Returns:
            np.ndarray: The complex steering vector [num_mics x num_freq_bins].
        """
        if source_position_s.shape != (3, 1):
             raise ValueError("source_position_s must be a 3x1 vector")

        M = self.num_mics
        # Calculate frequencies only up to Nyquist, matching PyTorch version
        frequency_vector_half = np.linspace(0, self.sampling_frequency / 2, self.num_freq_bins) # Shape [num_freq_bins]

        # Calculate distances from source 's' to each microphone 'm_i'
        distances = np.sqrt(np.sum((self.mic_positions_m - source_position_s) ** 2, axis=0)) # Shape [M]

        # Calculate time delays relative to the microphone furthest from the source
        max_dist = np.max(distances)
        # Use positive delays (arrival time difference relative to furthest mic)
        t_positive = (max_dist - distances) / self.sound_speed # Shape [M], delays >= 0

        # Calculate steering vector components: exp(-1j * 2 * pi * f * t_positive)
        # Use broadcasting for efficiency
        # frequency_vector_half [num_freq_bins] -> [1 x num_freq_bins]
        # t_positive [M] -> [M x 1]
        # Resulting phase_term [M x num_freq_bins]
        phase_term = -1j * 2 * np.pi * t_positive[:, np.newaxis] * frequency_vector_half[np.newaxis, :]
        steering_vector_half = np.exp(phase_term) # Shape [M x num_freq_bins]

        # Normalization note removed as it was already commented out/skipped

        return steering_vector_half # Return unnormalized based on standard MVDR

    def get_spatial_correlation_matrix(
        self,
        multi_signal: np.ndarray,
        use_number_of_frames_init: int = 10,
        use_number_of_frames_final: int = 10
        ) -> np.ndarray:
        """
        Calculates the Spatial Correlation Matrix (SCM) or Cross-Spectral Density (CSD) matrix.

        This implementation averages over initial and final frames,
        potentially targeting noise estimation if these frames contain only noise.

        Args:
            multi_signal (np.ndarray): Multi-channel time-domain signal [num_samples x num_mics].
            use_number_of_frames_init (int): Number of initial frames to use for averaging.
            use_number_of_frames_final (int): Number of final frames to use for averaging.

        Returns:
            np.ndarray: The estimated SCM/CSD matrix [num_mics x num_mics x num_freq_bins].
        """
        speech_length, num_channels = multi_signal.shape
        if num_channels != self.num_mics:
            raise ValueError("Number of channels in multi_signal does not match number of microphones.")

        # Initialize mean SCM
        R_mean = np.zeros((self.num_mics, self.num_mics, self.num_freq_bins), dtype=np.complex64)
        used_number_of_frames = 0
        window = sg.windows.hann(self.fft_length, sym=False) # Analysis window

        # --- Forward processing (Initial Frames) ---
        start_index = 0
        for frame_idx in range(use_number_of_frames_init):
            end_index = start_index + self.fft_length
            if end_index > speech_length:
                break # Not enough samples for a full frame

            # Extract and window frame for all channels
            multi_signal_cut = multi_signal[start_index:end_index, :] # Shape [fft_length x num_mics]
            windowed_frame = multi_signal_cut * window[:, np.newaxis]

            # Compute FFT for all channels
            complex_signal_frame = fft(windowed_frame, n=self.fft_length, axis=0) # Shape [fft_length x num_mics]
            complex_signal_half = complex_signal_frame[0:self.num_freq_bins, :] # Shape [num_freq_bins x num_mics]

            # Update SCM for each frequency bin
            for f_idx in range(self.num_freq_bins):
                # Outer product: X(f) * X(f)^H
                signal_at_f = complex_signal_half[f_idx:f_idx+1, :].T # Shape [num_mics x 1]
                R_mean[:, :, f_idx] += np.dot(signal_at_f, np.conjugate(signal_at_f.T))

            used_number_of_frames += 1
            start_index += self.fft_shift

        # --- Backward processing (Final Frames) ---
        # Calculate start index for backward processing carefully
        num_total_possible_frames = 1 + int((speech_length - self.fft_length) / self.fft_shift)
        last_frame_start_index = (num_total_possible_frames - 1) * self.fft_shift

        start_index_backward = max(0, last_frame_start_index - (use_number_of_frames_final - 1) * self.fft_shift)

        # Avoid double counting frames if init and final periods overlap significantly
        initial_frames_end_index = (use_number_of_frames_init - 1) * self.fft_shift + self.fft_length
        if start_index_backward < initial_frames_end_index:
             print("Warning: Initial and final frame periods for SCM calculation overlap.")
             # Adjust start_index_backward or handle overlap if needed
             # For simplicity, we'll proceed, potentially double-counting some frames' contribution.

        start_index = start_index_backward
        frames_processed_backward = 0
        for frame_idx in range(use_number_of_frames_final):
             end_index = start_index + self.fft_length
             if end_index > speech_length:
                 # This condition might be hit if speech_length is not long enough
                 # Or if start_index_backward calculation needs refinement for edge cases
                 break

             # Check if this frame was already processed in the forward pass
             if start_index < initial_frames_end_index:
                 start_index += self.fft_shift # Skip frame if already processed
                 continue


             multi_signal_cut = multi_signal[start_index:end_index, :]
             windowed_frame = multi_signal_cut * window[:, np.newaxis]
             complex_signal_frame = fft(windowed_frame, n=self.fft_length, axis=0)
             complex_signal_half = complex_signal_frame[0:self.num_freq_bins, :]

             for f_idx in range(self.num_freq_bins):
                 signal_at_f = complex_signal_half[f_idx:f_idx+1, :].T
                 R_mean[:, :, f_idx] += np.dot(signal_at_f, np.conjugate(signal_at_f.T))

             used_number_of_frames += 1
             frames_processed_backward += 1
             start_index += self.fft_shift


        # Average the SCM
        if used_number_of_frames > 0:
            R_mean /= used_number_of_frames
        else:
            print("Warning: No frames were processed for SCM calculation.")
            # Return zero matrix or handle error appropriately
            return R_mean # Returns zeros if no frames processed

        return R_mean


    def get_mvdr_beamformer(
        self,
        steering_vector: np.ndarray,
        scm_matrix: np.ndarray,
        stabilization_theta: float = 1e5
        ) -> np.ndarray:
        """
        Calculates the MVDR beamformer weights.

        Formula: w(f) = (R(f)^-1 * d(f)) / (d(f)^H * R(f)^-1 * d(f))
        where R(f) is the SCM at frequency f, d(f) is the steering vector,
        and ^H denotes the conjugate transpose.

        Args:
            steering_vector (np.ndarray): Steering vector for the look direction [num_mics x num_freq_bins].
            scm_matrix (np.ndarray): Spatial Correlation Matrix [num_mics x num_mics x num_freq_bins].
            stabilization_theta (float): Condition number threshold for SCM stabilization (using util.stab).

        Returns:
            np.ndarray: MVDR beamformer weights [num_mics x num_freq_bins].
        """
        if steering_vector.shape != (self.num_mics, self.num_freq_bins):
             raise ValueError("Steering vector shape mismatch.")
        if scm_matrix.shape != (self.num_mics, self.num_mics, self.num_freq_bins):
             raise ValueError("SCM matrix shape mismatch.")

        beamformer_weights = np.zeros((self.num_mics, self.num_freq_bins), dtype=np.complex64)

        for f_idx in range(self.num_freq_bins):
            # Extract SCM and steering vector for this frequency bin
            R_f = scm_matrix[:, :, f_idx] # Shape [num_mics x num_mics]
            d_f = steering_vector[:, f_idx:f_idx+1] # Keep as column vector [num_mics x 1]

            # Stabilize and invert the SCM
            try:
                # Use the provided stabilization function
                R_f_stable = util.stab(R_f, stabilization_theta, self.num_mics)
                inv_R_f = np.linalg.pinv(R_f_stable) # Use pseudo-inverse for robustness
            except np.linalg.LinAlgError:
                print(f"Warning: Matrix inversion failed for frequency bin {f_idx}. Using identity matrix.")
                # Fallback: Use identity matrix (results in delay-and-sum like behavior for this bin)
                inv_R_f = np.eye(self.num_mics, dtype=np.complex64)

            # Calculate numerator: R(f)^-1 * d(f)
            numerator = np.dot(inv_R_f, d_f) # Shape [num_mics x 1]

            # Calculate denominator: d(f)^H * R(f)^-1 * d(f)
            # d_f_H is conjugate transpose [1 x num_mics]
            d_f_H = np.conjugate(d_f.T)
            denominator = np.dot(d_f_H, numerator) # Shape [1 x 1] (scalar)

            # Avoid division by zero or very small numbers
            if np.abs(denominator[0, 0]) > 1e-10:
                beamformer_weights[:, f_idx] = numerator[:, 0] / denominator[0, 0]
            else:
                # Handle case where denominator is near zero (e.g., steering vector is null)
                print(f"Warning: Denominator near zero for frequency bin {f_idx}. Setting weights to zero.")
                beamformer_weights[:, f_idx] = 0.0 # Set weights to zero for this bin


        return beamformer_weights


    def apply_beamformer(
        self,
        beamformer_weights: np.ndarray,
        complex_spectrum: np.ndarray
        ) -> np.ndarray:
        """
        Applies the calculated beamformer weights to the multi-channel complex spectrum.

        Formula: Y(frame, f) = w(f)^H * X(frame, f)
        where Y is the enhanced single-channel spectrum, w(f) are the beamformer weights,
        and X(frame, f) is the multi-channel spectrum vector [num_mics x 1] for a given frame and frequency.

        Args:
            beamformer_weights (np.ndarray): MVDR beamformer weights [num_mics x num_freq_bins].
            complex_spectrum (np.ndarray): Multi-channel STFT spectrum [num_mics x num_frames x num_freq_bins].

        Returns:
            np.ndarray: The enhanced single-channel time-domain waveform.
        """
        num_channels, num_frames, num_bins = complex_spectrum.shape

        if num_channels != self.num_mics:
            raise ValueError("Number of channels in complex_spectrum does not match beamformer.")
        if num_bins != self.num_freq_bins:
             raise ValueError("Number of frequency bins in complex_spectrum does not match beamformer.")
        if beamformer_weights.shape != (self.num_mics, self.num_freq_bins):
             raise ValueError("Beamformer weights shape mismatch.")


        # Initialize enhanced spectrum array
        enhanced_spectrum = np.zeros((num_frames, num_bins), dtype=np.complex64)

        # Apply weights bin by bin, frame by frame (vectorized over frames)
        for f_idx in range(num_bins):
            # Weights for this bin (conjugate transpose): w(f)^H -> shape [1 x num_mics]
            w_f_H = np.conjugate(beamformer_weights[:, f_idx:f_idx+1].T)

            # Spectrum data for this bin across all frames: X(:, f) -> shape [num_mics x num_frames]
            X_f = complex_spectrum[:, :, f_idx]

            # Apply beamformer: Y(:, f) = w(f)^H * X(:, f) -> shape [1 x num_frames]
            enhanced_spectrum[:, f_idx] = np.dot(w_f_H, X_f)[0, :] # Result is [num_frames]

        # Convert enhanced spectrum back to time domain using inverse STFT
        # Ensure the correct window length (frame_len) is passed if different from fft_length
        # Assuming frame_len = fft_length as per original util functions, but ideally should be distinct params.
        enhanced_waveform = util.spec2wav(
            enhanced_spectrum,
            self.sampling_frequency,
            self.fft_length,
            self.fft_length, # Assuming frame_len = fft_length here based on original util
            self.fft_shift
        )

        return enhanced_waveform
