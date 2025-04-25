import numpy as np


class MicArray:
    """Creates an instance meant to represent an evenly spaced linear mic array.
    Consider that y values of all mics will be the same."""
    def __init__(self, numMicrophones, micSpacing):
        self.numMicrophones = numMicrophones
        self.micSpacing = micSpacing
        endX = (self.numMicrophones) * micSpacing
        micXLocations = np.arange(start=0, stop=endX, step=self.micSpacing)
        micYLocations = np.linspace(start=0, stop=0, num=self.numMicrophones)
        self.micLocations = np.array([micXLocations, micYLocations])

    def getMicLocations(self):
        return self.micLocations
