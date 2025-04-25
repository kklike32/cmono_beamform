import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

def load_audio(file_path):
    """
    Load audio file and return time series and sampling rate
    """
    sr, audio = wav.read(file_path)
    # Convert to float and normalize
    audio = audio.astype(float) / np.max(np.abs(audio))
    return audio, sr

def calculate_spectrogram(audio, sr, nperseg=256):
    """
    Generate spectrogram from audio time series
    """
    f, t, Sxx = signal.spectrogram(audio, fs=sr, nperseg=nperseg)
    return f, t, 10 * np.log10(Sxx)

def calculate_snir(original_audio, beamformed_audio, window_size=1024):
    """
    Calculate Signal-to-Noise plus Interference Ratio (SNIR)
    """
    def estimate_snir(signal_data):
        # Split signal into windows
        windows = np.array([signal_data[i:i+window_size] for i in range(0, len(signal_data)-window_size+1, window_size)])
        
        # Calculate power for each window
        window_power = np.mean(windows**2, axis=1)
        
        # Signal power (max power window)
        signal_power = np.max(window_power)
        
        # Noise + Interference power (average of other windows)
        noise_interference_power = np.mean(window_power[window_power < signal_power])
        
        # SNIR calculation
        snir = 10 * np.log10(signal_power / (noise_interference_power + 1e-10))
        #snir = signal_power/noise_interference_power
        return snir

    original_snir = estimate_snir(original_audio)
    beamformed_snir = estimate_snir(beamformed_audio)
    snir_improvement = beamformed_snir - original_snir

    return original_snir, beamformed_snir, snir_improvement

def plot_spectrograms(original_audio, beamformed_audio, sr):
    """
    Plot spectrograms for original and beamformed signals
    """
    plt.figure(figsize=(12, 6))
    
    # Original signal spectrogram
    plt.subplot(1, 2, 1)
    f, t, Sxx_orig = calculate_spectrogram(original_audio, sr)
    plt.pcolormesh(t, f, Sxx_orig, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Original Signal Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    
    # Beamformed signal spectrogram
    plt.subplot(1, 2, 2)
    f, t, Sxx_beam = calculate_spectrogram(beamformed_audio, sr)
    plt.pcolormesh(t, f, Sxx_beam, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Beamformed Signal Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    
    plt.tight_layout()
    plt.show()

def main(original_file_path, beamformed_file_path):
    # Load audio files
    original_audio, sr = load_audio(original_file_path)
    beamformed_audio, _ = load_audio(beamformed_file_path)

    # Plot spectrograms
    #plot_spectrograms(original_audio, beamformed_audio, sr)

    # Calculate SNIR
    original_snir, beamformed_snir, snir_improvement = calculate_snir(original_audio, beamformed_audio)

    # Print SNIR results
    print(f"Original Signal SNIR: {original_snir:.2f} dB")
    print(f"Beamformed Signal SNIR: {beamformed_snir:.2f} dB")
    print(f"SNIR Improvement: {snir_improvement:.2f} dB")

# Usage example (replace with your actual file paths)



main('angular_sweep_audio.wav', 'enhanced3.wav')