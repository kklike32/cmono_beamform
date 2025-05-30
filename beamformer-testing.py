import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm

# 1 ─ Array + constants
fs, dur = 44_100, 1.0
c, M, d = 343.0, 8, 0.02
mic_pos = np.arange(M) * d
noise_sigma = 0.5

# 2 ─ Scenario angles
theta_src  = -20      # desired source
theta_int  =  45      # interferer
theta_look = -20      # where MVDR is steered

# 3 ─ Tone list
freqs  = [100, 890, 1680, 2469, 3260, 4050, 4840, 5630, 6420, 7210, 8000]
colors = cm.rainbow(np.linspace(0, 1, len(freqs)))
angles = np.arange(-90, 91)

def steering_vector(f_hz, angle_deg):
    θ = np.deg2rad(angle_deg)
    λ = c / f_hz
    return np.exp(-1j * 2 * np.pi * mic_pos * np.sin(θ) / λ)

def estimate_cov(X):
    return (X @ X.conj().T) / X.shape[1]

def mvdr_weights(R, a):
    δ = 1e-6 * np.trace(R) / R.shape[0]
    w = np.linalg.inv(R + δ*np.eye(M)) @ a
    return w / (a.conj().T @ np.linalg.inv(R + δ*np.eye(M)) @ a)

def snr_db(sig, noise): return 10*np.log10(np.var(sig)/np.var(noise))

plt.figure(figsize=(10,6))

for f0, color in zip(freqs, colors):
    t = np.linspace(0, dur, int(fs*dur), endpoint=False)

    # --- build desired + interferer waveforms ---
    sig_des = np.sin(2*np.pi*f0*t)
    sig_int = 10**(10/20) * np.sin(2*np.pi*f0*t + np.pi/3)  # +10 dB, phase-offset

    # apply geometric delays
    def delayed(sig, angle):
        delays = mic_pos * np.sin(np.deg2rad(angle)) / c
        return np.vstack([np.roll(sig, int(round(τ*fs))) for τ in delays])

    X_des = delayed(sig_des, theta_src)
    X_int = delayed(sig_int, theta_int)

    # add uncorrelated sensor noise
    X = X_des + X_int + noise_sigma*np.random.randn(M, t.size)

    # --- MVDR design steered to desired angle ---
    a_look = steering_vector(f0, theta_look)
    R      = estimate_cov(X)
    w      = mvdr_weights(R, a_look)

    # SNR metrics
    y_sig  = w.conj().T @ X_des
    y_int  = w.conj().T @ X_int
    y_noi  = w.conj().T @ (X - X_des - X_int)
    input_sir  = snr_db(np.sum(X_des,0), np.sum(X_int,0))     # crude array SIR
    output_sir = snr_db(y_sig, y_int)
    print(f"{f0:5d} Hz | Array SIR: {input_sir:6.1f} dB → Output SIR: {output_sir:6.1f} dB")

    # Beampattern
    bp = [20*np.log10(abs(w.conj().T @ steering_vector(f0, θ))) for θ in angles]
    bp -= np.max(bp)
    plt.plot(angles, bp, color=color, label=f"{f0}")

plt.title(f"MVDR beampattern – look = {theta_look}°, desired = {theta_src}°, interferer = {theta_int}°")
plt.xlabel("Angle (deg)"); plt.ylabel("Normalised (dB)")
plt.grid(True); plt.ylim(-40,3); plt.xlim(-90,90)
plt.legend(title="Frequency (Hz)", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout(rect=[0,0,0.85,1]); plt.show()
