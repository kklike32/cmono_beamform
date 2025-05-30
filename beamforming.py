import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ------------------------------------------------------------------
# 1.  Steering-vector utilities
# ------------------------------------------------------------------
def steering_vector(freq_hz, angle_deg, mic_pos, c=343.0):
    θ = np.deg2rad(angle_deg)
    λ = c / freq_hz
    return np.exp(-1j * 2 * np.pi * mic_pos * np.sin(θ) / λ)

# ------------------------------------------------------------------
# 2.  Covariance + MVDR helper
# ------------------------------------------------------------------
def estimate_cov(X):
    return (X @ X.conj().T) / X.shape[1]

def mvdr_weights(R, a):
    δ = 1e-6 * np.trace(R) / R.shape[0]          # diagonal loading
    R_inv = np.linalg.inv(R + δ * np.eye(R.shape[0]))
    w = R_inv @ a
    return w / (a.conj().T @ R_inv @ a)

# ------------------------------------------------------------------
# 3.  Experiment settings
# ------------------------------------------------------------------
fs, dur     = 44_100, 1.0         # sample rate, seconds
c, M, d     = 343.0, 8, 0.02      # 8 mics, 2 cm spacing
mic_pos     = np.arange(M) * d
theta_src   = 0                   # broadside
#theta_src = -20 #stress test #2
#theta_int = 45 #interferer tone: stress test #2
#A_int_db = +10 #10 dB louder than desired: stress test #2

#theta_look = 30 #stress test #1

noise_sigma = 0.5 # adjust for SNR

# Frequencies to plot
freqs = [100, 890, 1680, 2469, 3260, 4050, 4840, 5630, 6420, 7210, 8000]
colors = cm.rainbow(np.linspace(0, 1, len(freqs)))

plt.figure(figsize=(10, 6))
angles = np.arange(-90, 91)

def snr_db(signal, noise):
    return 10*np.log10(np.var(signal)/np.var(noise))

for idx, (f0, color) in enumerate(zip(freqs, colors)):
    # Generate clean source and delayed multichannel data
    t   = np.linspace(0, dur, int(fs*dur), endpoint=False)
    sig = np.sin(2*np.pi*f0*t)
    delays = mic_pos * np.sin(np.deg2rad(theta_src)) / c
    X_sig  = np.vstack([np.roll(sig, int(round(τ*fs))) for τ in delays])
    sensor_noise = noise_sigma * np.random.randn(*X_sig.shape)
    X = X_sig + sensor_noise

    # MVDR design
    R     = estimate_cov(X)
    #a_look = steering_vector(f0, theta_look, mic_pos)   #stress test #1
    a_look = steering_vector(f0, theta_src, mic_pos)
    w     = mvdr_weights(R, a_look)

    # SNR calculation
    input_snr  = snr_db(sig, sensor_noise[0])
    output_snr = snr_db(w.conj().T @ X_sig, w.conj().T @ sensor_noise)
    print(f"Freq {f0:5d} Hz | Input SNR: {input_snr:6.2f} dB | Output SNR: {output_snr:6.2f} dB | Improvement: {output_snr - input_snr:6.2f} dB")

    # Beampattern
    bp = []
    for θ in angles:
        a_scan = steering_vector(f0, θ, mic_pos)
        bp.append(20*np.log10(np.abs(w.conj().T @ a_scan)))
    bp = np.array(bp) - np.max(bp)
    plt.plot(angles, bp, color=color, label=f'{f0}')

plt.title('MVDR Beampattern (Broadside Steering)')
plt.xlabel('Angle (deg)')
plt.ylabel('Normalized (dB)')
plt.grid(True)
plt.ylim(-40, 3)
plt.xlim(-90, 90)
plt.legend(title='Frequency (Hz)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
