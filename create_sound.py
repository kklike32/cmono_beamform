import matlab.engine  #matlab engine does not work  - use your converted matlab code
import numpy as np
from scipy.io.wavfile import write
import sys
# setting path
sys.path.append('..')
from settings import config
from mic_array import MicArray
import matplotlib.pyplot as plt

colors = ["red","green","blue","orange","purple","yellow","gray","pink"]

class SoundSynthesizer:
    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(r'C:/Program Files/MATLAB/R2023a')
        #C:\Program Files\MATLAB\R2023a

    def makeMonoAudio(self,frequency:int,writeFile:bool=False):
        x =self.eng.genSinWave(frequency)
        result = np.array(x)
        if writeFile:
            write("updated_sin1k.wav",config["SAMPLING_FREQUENCY"],result.astype(np.float32))
        return x#result
    
    def makeSingleSamplePulse(self,fs,duration):
        x = self.eng.genPulse(fs,duration)
        result = np.array(x)
        return x

    def delayMonoAcrossChannels(self,monoAudio,steeringAngle:float,numMics:int,micSeparation:float,fs:int):
        result = self.eng.delayMaker(monoAudio,steeringAngle,numMics,micSeparation,fs)
        return result

    def delayMonoAcrossChannelsF(self,monoAudio,steeringAngle:float,numMics:int,micSeparation:float,fs:int):
        result = self.eng.delayMakerF(monoAudio,steeringAngle,numMics,micSeparation,fs)
        return result

    def delayAcrossChannelsPy(self,monoAudio,steeringAngle,numMics,micSeparation,fs):
        """Delays mono channel audio across @numMics channels in the time domain. Accepts steeringAngle in degrees, not radians."""
        if np.shape(monoAudio)[0] == 1:
            monoAudio = monoAudio.T
        if numMics == 0:
            raise Exception("Must have >0 mic!")
        
        tdelay = micSeparation/config["SPEED_OF_SOUND"] * np.sin(np.radians(steeringAngle))
        ydelay = np.zeros((len(monoAudio),numMics))#np.copy(monoAudio)
        ydelay[:,0:1] = np.copy(monoAudio)
        for i in range(1,numMics):
            idelay = int(np.round(tdelay*i*fs) + 1)
            if idelay>0:
                ydelay[idelay-1:,i:i+1] = monoAudio[:len(monoAudio)-idelay+1]
            elif idelay==0:
                ydelay[idelay:,i:i+1] = monoAudio[:len(monoAudio)-idelay+1]
            else:
                ydelay[:len(monoAudio)-np.abs(idelay)+1,i:i+1] = monoAudio[np.abs(idelay)-1:]

        return ydelay

    def createDelayVector(self,speedOfSound,incidentAngleOfPlaneWaveAzimuth,numMics,micSeparation):
        self.micArray = MicArray(numMics,micSeparation)
        delay = np.linspace(0, 0, num=self.micArray.numMicrophones)
        for i in range(self.micArray.numMicrophones):
            delay[i] = (
                self.micArray.micLocations[0, i]
                * np.cos((np.pi / 2) - incidentAngleOfPlaneWaveAzimuth)
            ) / speedOfSound
        return delay 
    
    def set_steering_vector(self, delay,fft_length,fs):
            """Cochlearity's unique steering vector calculation algorithm. 
            We do this using azimuth i.e non elevation angle and mic array locations to calculate steering vector.
            Use the formula e^(j w t) where w is frequency of incoming plane wave, t is delay at that mic """
            frequency_vector = np.linspace(0, fs//2, fft_length//2)
            steering_vector = np.ones(
                (len(frequency_vector), self.micArray.numMicrophones),
                dtype=np.complex64,
            )
            radians = np.ones(
                (len(frequency_vector), self.micArray.numMicrophones),
                dtype=np.complex64,
            )
            for f, frequency in enumerate(frequency_vector):
                # [f:f+1,:] gives you the entire row of microphones
                for m in range(self.micArray.numMicrophones):
                    steering_vector[f : f + 1, m] = np.exp(
                        -1j * 2 * np.pi * frequency * delay[m]
                    )
                    radians[f : f + 1, m] = 2 * np.pi * frequency * delay[m]
            # apply hermitian
            steering_vector = np.conjugate(steering_vector).T
            return steering_vector

    def delayAcrossChannelsPyFreq(self,monoAudio,steeringAngle, numMics, micSeparation,fs):
        """Delays mono channel audio across @numMics channels in the frequency domain. steeringAngle must be in degrees, not radians"""
        if np.shape(monoAudio)[0] == 1:
            monoAudio = monoAudio.T
        if numMics == 0:
            raise Exception("Must have >0 mic!")
        steeringAngle = -1 * steeringAngle #have to do this as steering calcs later in this function steer exactly opposite.
        #fft monoAudio
        monoAudioF = np.fft.rfft(monoAudio,axis=0)
        #create ydelay matrix
        ydelay = np.zeros((len(monoAudioF),numMics),dtype=np.complex64)
        for m in range(numMics):
            ydelay[:,m:m+1] = monoAudioF
        angleRad = np.radians(steeringAngle)
        #elementwise multiply with phase shifts after getting delays
        delay = self.createDelayVector(config["SPEED_OF_SOUND"],angleRad,numMics,micSeparation)
        steering_vector = self.set_steering_vector(delay,len(monoAudio),fs)
        ydelay[:len(monoAudioF)-1,:] = np.multiply(ydelay[:len(monoAudioF)-1,:],steering_vector.T) 
        #ifft 8 channel result 
        result = np.fft.irfft(ydelay,axis=0)
        
        return result

if __name__ == "__main__":
    ss = SoundSynthesizer()
    monoAudio = ss.makeMonoAudio(1000,False)
    print(np.shape(np.array(monoAudio)))
    #result = ss.delayMonoAcrossChannelsF(monoAudio,45.0,8,config["MIC_SPACING"],config["SAMPLING_FREQUENCY"])
    #print((np.array(result)))
    #print(ss.delayAcrossChannelsPy(np.array(monoAudio),-60.0,8,config["MIC_SPACING"],config["SAMPLING_FREQUENCY"]))
    write("test.wav",48000,(32767*ss.delayAcrossChannelsPyFreq(np.array(monoAudio),50,8,0.0167,48000)).astype(np.int16))
    ss.delayAcrossChannelsPy(np.array(monoAudio),50,8,0.0167,48000)