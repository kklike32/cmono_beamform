import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft
import soundfile as sf

def make_mono_audio(frequency: int, write_file: bool = False, duration: float = 1.0, sampling_rate: int = 44100):
    """
    Generate a mono sine wave audio signal.

    :param frequency: Frequency of the sine wave in Hz.
    :param write_file: If True, saves the generated audio as a WAV file.
    :param duration: Duration of the generated sine wave in seconds (default: 1.0s).
    :param sampling_rate: Sampling rate in Hz (default: 44100 Hz).
    :return: NumPy array containing the generated sine wave.
    """
    noise = np.random.normal(0, 0.1, int(sampling_rate * duration))
    
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    result = np.sin(2 * np.pi * frequency * t)
    result += noise

    if write_file:
        write("updated_sin1k.wav", sampling_rate, result.astype(np.float32))

    return result

def create_delay_vector(speed_of_sound, angle_rad, num_mics, mic_separation):
    """
    Create a delay vector for microphone array beamforming.

    :param speed_of_sound: Speed of sound in m/s.
    :param angle_rad: Steering angle in radians.
    :param num_mics: Number of microphones in the array.
    :param mic_separation: Distance between adjacent microphones in meters.
    :return: Delay vector for each microphone.
    """
    mic_positions = np.arange(num_mics) * mic_separation
    delay_vector = -mic_positions * np.sin(angle_rad) / speed_of_sound
    return delay_vector

def set_steering_vector(delay_vector, signal_length, fs):
    """
    Generate a steering vector for phase shifting.

    :param delay_vector: Delays for each microphone.
    :param signal_length: Length of the mono audio signal.
    :param fs: Sampling frequency in Hz.
    :return: Steering vector for phase shifting.
    """
    freqs = np.fft.rfftfreq(signal_length, d=1/fs)
    steering_vector = np.exp(-1j * 2 * np.pi * freqs[:, np.newaxis] * delay_vector)
    return steering_vector

def delay_across_channels_py_freq(mono_audio, steering_angle, num_mics, mic_separation, fs, speed_of_sound=343.0):
    """
    Delays a mono audio signal across multiple microphones in the frequency domain.

    :param mono_audio: Input mono audio signal (NumPy array).
    :param steering_angle: Desired steering angle in degrees.
    :param num_mics: Number of microphones in the array.
    :param mic_separation: Distance between adjacent microphones in meters.
    :param fs: Sampling frequency in Hz.
    :param speed_of_sound: Speed of sound in m/s (default: 343 m/s).
    :return: Delayed multi-channel audio signal (NumPy array).
    """
    if mono_audio.ndim == 1:
        mono_audio = mono_audio[:, np.newaxis]  # Ensure column vector
    if num_mics <= 0:
        raise ValueError("Number of microphones must be greater than 0!")

    steering_angle = -steering_angle  # Reverse angle for correct steering calculations

    # FFT of the mono audio
    mono_audio_f = np.fft.rfft(mono_audio, axis=0)

    # Create an empty ydelay matrix
    ydelay = np.tile(mono_audio_f, (1, num_mics))  # Replicate signal across microphones

    # Convert angle to radians
    angle_rad = np.radians(steering_angle)

    # Compute delays and steering vector
    delay = create_delay_vector(speed_of_sound, angle_rad, num_mics, mic_separation)
    steering_vector = set_steering_vector(delay, len(mono_audio), fs)

    # Apply phase shifts (fixing broadcasting issue)
    ydelay *= steering_vector  # No transposition needed

    # Inverse FFT to get the time-domain signal
    result = np.fft.irfft(ydelay, axis=0)

    return result

def create_steering_vector(freqs, mic_positions, angle_deg, c=343.0):
    """
    Create a steering vector for the given frequency bins and microphone positions.
    
    :param freqs: Array of frequency bins
    :param mic_positions: Array of microphone positions
    :param angle_deg: Steering angle in degrees
    :param c: Speed of sound in m/s
    :return: Steering vector of shape (F, M) where F is number of frequency bins and M is number of microphones
    """
    angle_rad = np.deg2rad(angle_deg)
    return np.exp(-1j * 2 * np.pi * freqs[:, None] * mic_positions[None, :] * np.cos(angle_rad) / c)

def estimate_noise_covariance(X, noise_frames=10):
    """
    Estimate the noise covariance matrix from the first few frames of the signal.
    
    :param X: STFT of the signal, shape (M, F, T)
    :param noise_frames: Number of frames to use for noise estimation
    :return: Noise covariance matrix of shape (F, M, M)
    """
    X_noise = X[:, :, :noise_frames]
    Rn = np.einsum("mft,nft->fmn", X_noise, np.conj(X_noise)) / noise_frames
    # Add small regularization
    for k in range(Rn.shape[0]):
        Rn[k] += 1e-6 * np.trace(Rn[k]) * np.eye(Rn.shape[1]) / Rn.shape[1]
    return Rn

def compute_mvdr_weights(steering_vector, Rn, reg=1e-6):
    """
    Compute MVDR beamforming weights.
    
    :param steering_vector: Steering vector of shape (F, M)
    :param Rn: Noise covariance matrix of shape (F, M, M)
    :param reg: Regularization parameter
    :return: Beamforming weights of shape (F, M)
    """
    w = np.empty_like(steering_vector, dtype=np.complex128)
    for k in range(len(steering_vector)):
        R_inv_a = np.linalg.solve(Rn[k] + reg * np.eye(Rn.shape[1]), steering_vector[k])
        denom = np.conj(steering_vector[k]).T @ R_inv_a
        w[k] = R_inv_a / denom
    return w

# ------------------------ Array & signal parameters ------------------------
freq = 12000  # Frequency of the test signal
c = 343.0     # Speed of sound (m/s)
M = 8         # Number of microphones
d = 0.01      # Microphone spacing (m)
win_len = 1024
hop = win_len // 2
window = "hann"
fs = 44100

# Create microphone positions
mic_positions = np.arange(M) * d

# Generate test signal
mono_audio = make_mono_audio(freq, False, 5, fs)

# Initialize arrays for polar plot
angleArr = []
logOutputArr = []
angleRadArr = []

maxAngle, maxVal = 0, -300
N = 360  # Number of angles to test

# Perform beamforming for each angle
for a in range(N):
    angle = 360 * a / (N - 1)
    angleRad = angle * (np.pi/180)
    
    # Create delayed signals for each microphone
    audio = delay_across_channels_py_freq(mono_audio, angle, M, d, fs, c)
    
    # Compute STFT
    F, T, X = stft(audio.T, fs=fs, window=window, nperseg=win_len, noverlap=hop, axis=-1)
    
    # Get frequency bins
    freqs = np.fft.rfftfreq(win_len, 1/fs)
    
    # Create steering vector
    steering_vector = create_steering_vector(freqs, mic_positions, angle)
    
    # Estimate noise covariance
    Rn = estimate_noise_covariance(X)
    
    # Compute MVDR weights
    w = compute_mvdr_weights(steering_vector, Rn)
    
    # Apply beamforming
    X = X.transpose(1, 0, 2)  # Reshape to (F, M, T)
    Y = np.sum(np.conj(w)[:, :, None] * X, axis=1)  # Beamformed output (F, T)
    
    # Reconstruct time domain signal
    _, y_time = istft(Y, fs=fs, window=window, nperseg=win_len, noverlap=hop)
    
    # Compute output power
    output_power = np.max(np.abs(y_time))
    if maxVal < output_power:
        maxVal = output_power
        maxAngle = angle
    
    # Convert to dB
    logOutput = 20 * np.log10(output_power)
    
    # Store results
    angleRadArr.append(angleRad)
    angleArr.append(angle)
    logOutputArr.append(logOutput)

print(f"Maximum response at angle {maxAngle}° with value {maxVal}")

# Create polar plot
plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='polar')
ax.plot(angleRadArr, logOutputArr, 'b-', linewidth=2)

# Set the title and labels
plt.title('Beamformer Response Pattern', pad=20, size=14)
ax.set_theta_zero_location('N')  # Set 0 degrees to North
ax.set_theta_direction(-1)  # Set clockwise direction

# Add grid and improve its appearance
ax.grid(True, linestyle='--', alpha=0.7)

# Set the radial axis label
ax.set_ylabel('Magnitude (dB)', labelpad=20)

# Set the angle labels
ax.set_thetagrids(np.arange(0, 360, 45))

# Set the radial limits based on the data
rmin = min(logOutputArr)
rmax = max(logOutputArr)
ax.set_ylim(rmin - 5, rmax + 5)

# Add a legend
plt.legend(['Beamformer Response'], loc='upper right')

plt.tight_layout()
plt.show()
