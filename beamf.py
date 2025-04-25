import numpy as np
import soundfile as sf
from beamformer import util
from beamformer import minimum_variance_distortionless_response as mvdr
from scipy.io import wavfile
import pandas as pd 



SAMPLING_FREQUENCY = 44100
FFT_LENGTH = 512
FFT_SHIFT = 256
#ENHANCED_WAV_NAME = './output/enhanced_speech_mvdr.wav'
MIC_ANGLE_VECTOR = np.array([-30,30])
LOOK_DIRECTION = 90
#MIC_DIAMETER = 0.1
D = 0.01


look_set = pd.read_csv('ang_sweep_pos.csv')
look_set = look_set[['confidence','ellipse_angle']]

sr,multi_channels_data = wavfile.read('angular_sweep_audio.wav') 
print(multi_channels_data.shape)
seconds = int(len(multi_channels_data)/SAMPLING_FREQUENCY)

final_audio = []
for i in range(seconds):

    look_audio_section = multi_channels_data[int(i*SAMPLING_FREQUENCY):int((i+1)*SAMPLING_FREQUENCY),:]
    look_subset = look_set.iloc[i*300:(i+1)*300]
    lookangle = look_subset[look_subset['confidence']>0.6]['ellipse_angle'].mean()
    print(lookangle)

    #print(look_audio_section.shape)
    complex_spectrum, _ = util.get_3dim_spectrum_from_data(look_audio_section, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)

    mvdr_beamformer = mvdr.minimum_variance_distortioless_response(MIC_ANGLE_VECTOR, D, sampling_frequency=SAMPLING_FREQUENCY, fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)

    steering_vector = mvdr_beamformer.get_sterring_vector1(lookangle)

    spatial_correlation_matrix = mvdr_beamformer.get_spatial_correlation_matrix(look_audio_section)

    beamformer = mvdr_beamformer.get_mvdr_beamformer(steering_vector, spatial_correlation_matrix)

    enhanced_speech = mvdr_beamformer.apply_beamformer(beamformer, complex_spectrum)

    normalized_speech = enhanced_speech / (np.max(np.abs(enhanced_speech)) * 0.65)

    #print(normalized_speech.shape)

    final_audio.append(normalized_speech)

#print(np.shape(final_audio))
enhanced_speech_audio = np.array(final_audio)
enhanced_speech_audio = enhanced_speech_audio.ravel()
print(enhanced_speech_audio.shape)
sf.write('enhanced3.wav', enhanced_speech_audio, SAMPLING_FREQUENCY)
