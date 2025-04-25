import math
import numpy as np


def pol2cart(theta, rho, z):
    return (
        rho*math.cos(theta),
        rho*math.sin(theta),
        z
    )


def calculatePos3d(micSpacing, numMics, targetAngle):
    DISTANCE_FROM_MIC = 1.5  # this is an assumption

    # X axis is in line with the row of mics (on the azimuth plane)
    # y axis is perpendicular to the axis of the row of mics (on the azimuth plane)
    # z axis is vertical axis (should be set to 0)

    z = 0  # because we're on the same azimuth plane

    # calculate x,y,z source location for target source
    if targetAngle >= 0:  # we need all angles to be positive, clockwise angle as per pol2cart documentation
        sourceTheta = math.radians(targetAngle)
    else:
        sourceTheta = math.radians(targetAngle + 360)

    sourceRho = DISTANCE_FROM_MIC
    sourceX, sourceY, sourceZ = pol2cart(sourceTheta, sourceRho, z)
    sourceLoc = np.array([[sourceX], [sourceY], [sourceZ]])

    micLocs = np.zeros((3, int(numMics)))  # preload miclocs matrix with zeros

    # calculate and build micLocs x,y,z matrix
    micOverallSpacing = micSpacing * (numMics - 1)  # distance from left most mic to right most mic
    for i in range(numMics):
        micLocs[0, i] = 0  # because they all have the same relative x-axis position
        micLocs[1, i] = (i * micSpacing) + (-1 * micOverallSpacing / 2)  # position in meters on y-axis
        micLocs[2, i] = z  # elevation, which should be zero

    return (sourceLoc, micLocs)

