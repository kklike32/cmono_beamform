import numpy as np
from scipy.signal import stft, istft


def make_mono_audio(frequency: int, write_file: bool = False, duration: float = 1.0, sampling_rate: int = 44100):
    """
    Generate a mono sine wave audio signal.

    :param frequency: Frequency of the sine wave in Hz.
    :param write_file: If True, saves the generated audio as a WAV file.
    :param duration: Duration of the generated sine wave in seconds (default: 1.0s).
    :param sampling_rate: Sampling rate in Hz (default: 44100 Hz).
    :return: NumPy array containing the generated sine wave.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    result = np.sin(2 * np.pi * frequency * t)

    # if write_file:
    #     write("updated_sin1k.wav", sampling_rate, result.astype(np.float32))

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

signal_freq = [315,640,1250,2500,4000,6000]

for i in signal_freq:
    # ------------------------ Array & signal parameters ------------------------
    c               = 343.0                         # speed of sound (m/s)
    #fs, sig         = sf.read("8ch_recording.wav")  # sig → shape (samples , 8)
    M               = 8                # = 8
    win_len         = 1024                          # STFT window size
    hop             = win_len // 2
    window          = "ham"
    fs = 44100

    # ----------- ❶ steering vector for the desired look-direction --------------
    # Example: linear array, d = 4 cm spacing, plane-wave from broadside (0°)
    d               = 0.01
    angles_deg      = 45                           # target azimuth (change as needed)
    angles_rad      = np.deg2rad(angles_deg)
    mic_positions   = np.arange(M) * d              # x-coordinates along array


    f               = np.fft.rfftfreq(win_len, 1/fs)   # shape (F,)
    # steering vector a(f) → shape (F , M)
    #ster = np.exp(-1j * (2*np.pi*f[:,None]) * mic_positions[None,:] * np.cos(angles_rad) / c)

    ster = np.exp(-1j * 2*np.pi*f[:,None] * mic_positions[None,:]
                    * np.cos(np.deg2rad(angles_deg)) / c)

    mono_audio = make_mono_audio(i,False,5,fs)
    angleArr = []
    logOutputArr =[]
    angleRadArr = []

    maxAngle,maxVal = 0,-300
    for a in range(500):
        angle = -90 + 180 * (a/(500-1))
        angleRad = angle * (np.pi/180)

        #audio = ss.delayAcrossChannelsPyFreq(np.array(monoAudio),angle,8,config["MIC_SPACING"],config["SAMPLING_FREQUENCY"])
        audio = delay_across_channels_py_freq(mono_audio,angle,8,d,fs,343)
        F, T, X = stft(audio.T, fs=fs, window=window, nperseg=win_len, noverlap=hop, axis=-1)
        # X shape: (M , F , T)

        # ---- ❸ estimate spatial covariance of noise R_n(f)  ----------------------
        # Simple approach: use initial 0.5 s as “noise-only”; improve with VAD if needed
        # noise_samples   = int(0.5 * fs)
        # _, _, X_noise   = stft(audio[:noise_samples].T, fs=fs, window=window,
        #                     nperseg=win_len, noverlap=hop, axis=-1)
        # Rn = np.einsum("mft,nft->fmn", X_noise, np.conj(X_noise)) / X_noise.shape[-1]  # (F,M,M)

        Rn = np.tile(np.eye(M), (len(f), 1, 1))                       # (F , M , M) Covariance matrix

        # ---- MVDR weights --------------------------------------------------------
        w = np.empty_like(ster, dtype=np.complex128)
        for k in range(len(f)):
            R_inv_a = np.linalg.solve(Rn[k] + 1e-6*np.eye(M), ster[k])
            w[k]    = R_inv_a / (ster[k].conj() @ R_inv_a)
        # ❹ MVDR weights  w(f) = R_n⁻¹ a / (aᴴ R_n⁻¹ a)
        w = np.empty_like(ster, dtype=np.complex128)        # (F , M)
        X = X.transpose(1, 0, 2)

        for k in range(len(f)):
            R_inv_a          = np.linalg.solve(Rn[k] + 1e-6*np.eye(M), ster[k])   # regularised
            denom            = np.conj(ster[k]).T @ R_inv_a
            w[k]             = R_inv_a / denom

        # ---------------- ❺ apply beam-former & reconstruct -----------------------
        # Y = np.sum(np.conj(w[...,None]) * X, axis=0)     # (F , T)
        Y = np.sum(np.conj(w)[:, :, None] * X, axis=1)   # result (F , T)

        _, y_time = istft(Y, fs=fs, window=window, nperseg=win_len, noverlap=hop)
        
        normalized_speech =  np.abs(y_time)
        if maxVal < np.max(np.abs(normalized_speech)):
            maxVal = np.max(np.abs(normalized_speech))
            maxAngle = angle
        
        output = np.max(normalized_speech) 
        logOutput = 20 * np.log10(output) if 20 * np.log10(output) >= -50 else -50
        angleRadArr.append(angleRad)
        angleArr.append(angle)
        logOutputArr.append(logOutput)

    np.save(f'lee_res/angleRadArr_{i}.npy',angleRadArr)
    np.save(f'lee_res/logOutputArr_{i}.npy',logOutputArr)
    #print(a)