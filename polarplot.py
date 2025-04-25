import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from beamformer import util
from beamformer import minimum_variance_distortionless_response as mvdr
from scipy.io import wavfile
from calculatePos3d import calculatePos3d
import pandas as pd 



SAMPLING_FREQUENCY = 44100
FFT_LENGTH = 1024
FFT_SHIFT = 512
#ENHANCED_WAV_NAME = './output/enhanced_speech_mvdr.wav'
MIC_ANGLE_VECTOR = np.array([0,0,0,0,0,0,0,0])
LOOK_DIRECTION = 90
#MIC_DIAMETER = 0.1
c = 343
D = 0.01
M  = 8
theta = 0
s,m = calculatePos3d(D,M,theta)


# look_set = pd.read_csv('ang_sweep_pos.csv')
# look_set = look_set[['confidence','ellipse_angle']]

sr,multi_channels_data = wavfile.read('test_matlab30.wav') 
print(multi_channels_data.shape)
seconds = int(len(multi_channels_data)/SAMPLING_FREQUENCY)





angleArr = []
logOutputArr =[]
angleRadArr = []

maxAngle,maxVal = 0,-300
for a in range(500):
    angle = -90 + 180 * (a/(500-1))
    angleRad = angle * (np.pi/180)

    # audio = ss.delayAcrossChannelsPyFreq(np.array(monoAudio),angle,8,config["MIC_SPACING"],config["SAMPLING_FREQUENCY"])
    # ds.reset_audio(audio)
    # ds.set_steering_vector(delay)
    # beamformed_audio = ds.apply_steering_vector()
    # scaled_audio = beamformed _audio
    sr,scaled_audio = wavfile.read('enhanced30.wav') 


    if maxVal < np.max(np.abs(scaled_audio)):
        maxVal = np.max(np.abs(scaled_audio))
        maxAngle = angle
   
    output = np.max(scaled_audio) 
    logOutput = 20 * np.log10(output) if 20 * np.log10(output) >= -50 else -50
    angleRadArr.append(angleRad)
    angleArr.append(angle)
    logOutputArr.append(logOutput)

print(f"max angle was at {maxAngle} and value is {maxVal}")
#if config["POLAR"] == True:
plt.polar(angleRadArr,logOutputArr,"b-")
plt.show()
# else:
#     plt.xlabel("Incident angle")
#     plt.ylabel("Beamformer Response")
#     plt.plot(angleArr,logOutputArr,"b-")
#     plt.show()