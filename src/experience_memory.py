import numpy as np
import random

class ExperienceMemory(object):
    def __init__(self, capacity, stateLength):
        self.__capacity = capacity
        self.__usedCapacity = 0
        self.__stateLength = stateLength
        self.__writePosition = 0
        self.__writePositionReseted = False

        self.__ids = np.zeros((self.__capacity), dtype='uint64')
        self.__states = np.zeros((self.__capacity, self.__stateLength), dtype='float32')
        self.__actions = np.zeros((self.__capacity), dtype='uint8')
        self.__rewards = np.zeros((self.__capacity), dtype='float32')
        self.__nextStates = np.zeros((self.__capacity, self.__stateLength), dtype='float32')
        self.__nextStateIsTerminalStates = np.zeros((self.__capacity), dtype='bool')
        self.__sampleCounter = 0

    def store(self, state, action, reward, nextState, nextStateIsTerminalState):
        experienceId = self.__sampleCounter
        self.__ids[self.__writePosition] = experienceId
        self.__states[self.__writePosition] = state
        self.__actions[self.__writePosition] = action
        self.__rewards[self.__writePosition] = reward
        self.__nextStates[self.__writePosition] = nextState
        self.__nextStateIsTerminalStates[self.__writePosition] = nextStateIsTerminalState

        self.__writePosition += 1
        self.__sampleCounter += 1

        if not self.__writePositionReseted:
            self.__usedCapacity += 1

        if self.__writePosition == self.__capacity:
            self.__writePosition = 0
            self.__writePositionReseted = True

        return experienceId

    def size(self):
        return self.__usedCapacity

    def sample(self, numberOfSamples):
        if self.__usedCapacity < numberOfSamples:
            return None, None, None, None, None, None

        sampleIndex = random.sample(range(self.__usedCapacity), numberOfSamples)

        return self.__ids[sampleIndex], self.__states[sampleIndex], self.__actions[sampleIndex], self.__rewards[sampleIndex], self.__nextStates[sampleIndex], self.__nextStateIsTerminalStates[sampleIndex]
