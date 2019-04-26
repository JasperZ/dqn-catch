import numpy as np
import copy
from enum import Enum, unique

class Catch:
    def __init__(self, width, height, stateAsCoordinates=False, normalizeState=True):
        self.__width = width
        self.__height = height
        self.__stateAsCoordinates = stateAsCoordinates
        self.__normalizeState = normalizeState

        self.__position = [0, 0]
        self.__target = [0, 0]

        self.reset()

    def reset(self):
        self.__resetPosition()
        self.__resetTarget()

    def __resetPosition(self, avoidTarget=False):
        if avoidTarget:
            self.__position = self.__generateCoordinates(self.__target)
        else:
            self.__position = self.__generateCoordinates()

    def __resetTarget(self, avoidPosition=True):
        if avoidPosition:
            self.__target = self.__generateCoordinates(self.__position)
        else:
            self.__target = self.__generateCoordinates()



    def __generateCoordinates(self, coordinatesToAvoid=None):
        if coordinatesToAvoid == None:
            x = np.random.randint(0, self.__width)
            y = np.random.randint(0, self.__height)

            return [x, y]
        else:
            x = np.random.randint(0, self.__width)
            y = np.random.randint(0, self.__height)

            while [x, y] == coordinatesToAvoid:
                x = np.random.randint(0, self.__width)
                y = np.random.randint(0, self.__height)

            return [x, y]

    def __checkCoordinates(self, coordinatesToCheck):
        outOfBounds = False
        matchesTarget = False

        if coordinatesToCheck[0] < 0 or coordinatesToCheck[0] >= self.__width:
            outOfBounds = True

        if coordinatesToCheck[1] < 0 or coordinatesToCheck[1] >= self.__height:
            outOfBounds = True

        if coordinatesToCheck == self.__target:
            matchesTarget = True

        return (outOfBounds, matchesTarget)

    def __computeTempPosition(self, action):
        tmpPosition = copy.copy(self.__position)

        if action == Actions.LEFT:
            tmpPosition[0] -= 1
        elif action == Actions.RIGHT:
            tmpPosition[0] += 1
        elif action == Actions.UP:
            tmpPosition[1] -= 1
        elif action == Actions.DOWN:
            tmpPosition[1] += 1

        return tmpPosition

    def getState(self):
        if self.__stateAsCoordinates:
            return self.__generateCoordinateState()
        else:
            return self.__generateFieldState()

    def __generateCoordinateState(self):
        position = np.array(self.__position)
        target = np.array(self.__target)

        if self.__normalizeState:
            normalizationFactor = 1.0 / np.array([self.__width - 1, self.__height - 1])
            position = position * normalizationFactor
            target = target * normalizationFactor

        return np.concatenate((position, target))

    def __generateFieldState(self):
        field = np.zeros((self.__height, self.__width))

        field[self.__position[1]][self.__position[0]] = 1
        field[self.__target[1]][self.__target[0]] = 2

        if self.__normalizeState:
            field = field / np.max(field)

        return field.reshape((-1))

    def move(self, action):
        tmpPosition = self.__computeTempPosition(action)
        outOfBounds, matchesTarget = self.__checkCoordinates(tmpPosition)

        if not outOfBounds:
            self.__position = tmpPosition

        reward = -0.5

        if outOfBounds:
            reward = -1.0
        elif matchesTarget:
            reward = 1.0

        stateAfterAction = self.getState()

        return (reward, stateAfterAction, matchesTarget)

    def getNumberOfActions(self):
        return len(Actions)

    def getStateSize(self):
        return len(self.getState())


@unique
class Actions(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
