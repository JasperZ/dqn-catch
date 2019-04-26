import tensorflow as tf
import numpy as np
import argparse
import pathlib
import os

from catch import Catch, Actions
from experience_memory import ExperienceMemory

def buildDescriptiveString(experienceMemoryCapacity, minibatchSize, \
        gamma, epsilonStart, epsilonEnd, epsilonDecaySteps, learningRate, \
        stateAsCoordinates, stateNormalization, fieldWidth, fieldHeight, \
        useTargetNetwork, targetNetworkUpdateSteps):

    template = "lR={}_eD={}_sAC={}_sN={}_tN={}_tNUS={}_fW={}_fH={}_mB={}_mC={}_g={}_eS={}_eE={}"
    templateFilled = template.format(learningRate, epsilonDecaySteps, \
        stateAsCoordinates, stateNormalization, useTargetNetwork, \
        targetNetworkUpdateSteps, fieldWidth, fieldHeight, minibatchSize, \
        experienceMemoryCapacity, gamma, epsilonStart, epsilonEnd)

    return templateFilled

def saveModel(directory, step, session):
    filename = 'step={:010}.ckpt'.format(step)
    filepath = os.path.join(directory, filename)

    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    saver = tf.train.Saver()
    save_path = saver.save(session, filepath)

def createModel(stateSize, numberOfActions, isTargetNetwork=False):
    scope = "online"

    if isTargetNetwork:
        scope = "target"

    with tf.variable_scope(scope):
        input = tf.placeholder(shape=[None, stateSize], name="input", dtype=tf.float32)
        outputLabel = None

        with tf.name_scope("input_layer"):
            inputLayer = tf.layers.Dense(units=stateSize, use_bias=True)
            inputLayerOutput = inputLayer(input)
            inputLayerActivation = tf.nn.relu(inputLayerOutput)

            tf.summary.histogram("kernel", inputLayer.kernel)
            tf.summary.histogram("bias", inputLayer.bias)

            if not isTargetNetwork:
                tf.summary.histogram("output", inputLayerOutput)
                tf.summary.histogram("activations", inputLayerActivation)

        with tf.name_scope("hidden_layer_1"):
            hiddenLayer1 = tf.layers.Dense(units=128, use_bias=True)
            hiddenLayer1Output = hiddenLayer1(inputLayerActivation)
            hiddenLayer1Activation = tf.nn.relu(hiddenLayer1Output)

            tf.summary.histogram("kernel", hiddenLayer1.kernel)
            tf.summary.histogram("bias", hiddenLayer1.bias)

            if not isTargetNetwork:
                tf.summary.histogram("output", hiddenLayer1Output)
                tf.summary.histogram("activations", hiddenLayer1Activation)

        with tf.name_scope("hidden_layer_2"):
            hiddenLayer2 = tf.layers.Dense(units=64, use_bias=True)
            hiddenLayer2Output = hiddenLayer2(hiddenLayer1Activation)
            hiddenLayer2Activation = tf.nn.relu(hiddenLayer2Output)

            tf.summary.histogram("kernel", hiddenLayer2.kernel)
            tf.summary.histogram("bias", hiddenLayer2.bias)

            if not isTargetNetwork:
                tf.summary.histogram("output", hiddenLayer2Output)
                tf.summary.histogram("activations", hiddenLayer2Activation)

        with tf.name_scope("hidden_layer_3"):
            hiddenLayer3 = tf.layers.Dense(units=32, use_bias=True)
            hiddenLayer3Output = hiddenLayer3(hiddenLayer2Activation)
            hiddenLayer3Activation = tf.nn.relu(hiddenLayer3Output)

            tf.summary.histogram("kernel", hiddenLayer3.kernel)
            tf.summary.histogram("bias", hiddenLayer3.bias)

            if not isTargetNetwork:
                tf.summary.histogram("output", hiddenLayer3Output)
                tf.summary.histogram("activations", hiddenLayer3Activation)

        with tf.name_scope("hidden_layer_4"):
            hiddenLayer4 = tf.layers.Dense(units=16, use_bias=True)
            hiddenLayer4Output = hiddenLayer4(hiddenLayer3Activation)
            hiddenLayer4Activation = tf.nn.relu(hiddenLayer4Output)

            tf.summary.histogram("kernel", hiddenLayer4.kernel)
            tf.summary.histogram("bias", hiddenLayer4.bias)

            if not isTargetNetwork:
                tf.summary.histogram("output", hiddenLayer4Output)
                tf.summary.histogram("activations", hiddenLayer4Activation)

        with tf.name_scope("output_layer"):
            outputLayer = tf.layers.Dense(units=numberOfActions, use_bias=True)
            outputLayerOutput = outputLayer(hiddenLayer4Activation)

            tf.summary.histogram("kernel", outputLayer.kernel)
            tf.summary.histogram("bias", outputLayer.bias)

            if not isTargetNetwork:
                tf.summary.histogram("output", outputLayerOutput)

        output = outputLayerOutput

        if not isTargetNetwork:
            outputLabel = tf.placeholder(tf.float32, shape=[None, numberOfActions])
            tf.summary.histogram("output_label", outputLabel)

    mergedSummary = tf.summary.merge_all(scope=scope)

    return (input, output, outputLabel, mergedSummary)

def updateTargetNetwork(session, writer, targetSummary, step):
    updateWeights = [tf.assign(target, online) for (target, online) in  \
        zip(tf.trainable_variables("target"), tf.trainable_variables("online"))]

    session.run(updateWeights)
    targetSummaryResult = session.run(targetSummary)
    writer.add_summary(targetSummaryResult, step)

def main():
    STEP_LOG_RATE = 1000
    TENSORBOARD_ROOT_PATH = "tensorboard"
    CHECKPOINT_ROOT_PATH = "checkpoints_test"
    CHECKPOINTS_STEPS = 100000
    EXPERIENCE_MEMORY_CAPACITY = 6400000
    MINIBATCH_SIZE = 32
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY_STEPS = 1000000
    LEARNING_RATE = 0.0001
    FIELD_WIDTH = 5
    FIELD_HEIGHT = 5
    USE_TARGET_NETWORK = True
    TARGET_NETWORK_UPDATE_STEPS = 10000
    STATE_AS_COORDINATES = True
    STATE_NORMALISATION = True

    descriptiveString = buildDescriptiveString(EXPERIENCE_MEMORY_CAPACITY, \
        MINIBATCH_SIZE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS, \
        LEARNING_RATE, STATE_AS_COORDINATES, STATE_NORMALISATION, \
        FIELD_WIDTH, FIELD_HEIGHT, USE_TARGET_NETWORK, TARGET_NETWORK_UPDATE_STEPS)

    tensorboardDirectory = os.path.join(TENSORBOARD_ROOT_PATH, descriptiveString)
    checkpointDirectory = os.path.join(CHECKPOINT_ROOT_PATH, descriptiveString)

    # create catch environment
    catch = Catch(FIELD_WIDTH, FIELD_HEIGHT, STATE_AS_COORDINATES, STATE_NORMALISATION)
    numberOfActions = catch.getNumberOfActions()
    stateSize = catch.getStateSize()
    # create experience memory
    experienceMemory = ExperienceMemory(EXPERIENCE_MEMORY_CAPACITY, stateSize)

    ########################################################################################################################################################
    input, output, outputLabel, onlineSummary = createModel(stateSize, \
        numberOfActions, isTargetNetwork=False)

    if USE_TARGET_NETWORK:
        targetInput, targetOutput, _, targetSummary = createModel(stateSize, \
            numberOfActions, isTargetNetwork=True)

    with tf.name_scope("train"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
        loss = tf.losses.huber_loss(labels=outputLabel, predictions=output)
        train = optimizer.minimize(loss)

        tf.summary.scalar("loss", loss)

    episodicStepsSummary = tf.Summary()
    episodicRewardSummary = tf.Summary()
    explorationSummary = tf.Summary()
    experienceMemorySizeSummary = tf.Summary()

    episodicStepsSummary.value.add(tag="episodic_steps", simple_value=None)
    episodicRewardSummary.value.add(tag="episodic_reward", simple_value=None)
    explorationSummary.value.add(tag="exploration", simple_value=None)
    experienceMemorySizeSummary.value.add(tag="experience_memory_size", simple_value=None)

    trainSummary = tf.summary.merge_all(scope="train")

    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)

    writer = tf.summary.FileWriter(tensorboardDirectory, sess.graph)

    updateTargetNetwork(sess, writer, targetSummary, 0)
    ########################################################################################################################################################

    step = 0
    episode = 0
    epsilon = EPSILON_START

    while step < EPSILON_DECAY_STEPS:
        episode += 1

        catch.reset()
        state = catch.getState()
        done = False
        episodeReward = 0
        episodeSteps = 0

        while not done and step < EPSILON_DECAY_STEPS:
            step += 1
            # select next action
            if np.random.random() <= epsilon:
                actionNumber = np.random.randint(numberOfActions)
            else:
                prediction = sess.run(output, feed_dict={input: np.reshape(state, (-1, stateSize))})
                actionNumber = np.argmax(prediction[0])
            # convert action number to action
            action = list(Actions)[actionNumber]
            # execute selected action
            reward, nextState, done = catch.move(action)
            # store experience to memory
            experienceMemory.store(state, actionNumber, reward, nextState, done)
            # replace current state by next state
            state = nextState
            # replay experiences
            if experienceMemory.size() > MINIBATCH_SIZE:
                # sample from experience memory
                ids, states, actions, rewards, nextStates, nextStateTerminals = experienceMemory.sample(MINIBATCH_SIZE)

                if USE_TARGET_NETWORK:
                    statePredictions = sess.run(output, feed_dict={input: states})
                    nextStatePredictions = sess.run(targetOutput, feed_dict={targetInput: nextStates})
                else:
                    predictions = sess.run(output, feed_dict={input: np.concatenate((states, nextStates))})
                    statePredictions = predictions[:MINIBATCH_SIZE]
                    nextStatePredictions = predictions[MINIBATCH_SIZE:]

                statePredictions[np.arange(MINIBATCH_SIZE), actions] = \
                                rewards + np.invert(nextStateTerminals) * GAMMA * \
                                nextStatePredictions.max(axis=1)

                # update online network
                _, onlineSummaryResult, trainSummaryResult = sess.run([train, onlineSummary, trainSummary], feed_dict={input: states, outputLabel: statePredictions})
                # write summary
                if step % STEP_LOG_RATE == 0:
                    writer.add_summary(onlineSummaryResult, step)
                    writer.add_summary(trainSummaryResult, step)

            episodeReward += reward
            episodeSteps += 1
            # update target network
            if USE_TARGET_NETWORK and step % TARGET_NETWORK_UPDATE_STEPS == 0:
                updateTargetNetwork(sess, writer, targetSummary, step)
            # write exploration summary
            if step % STEP_LOG_RATE == 0:
                explorationSummary.value[0].simple_value = epsilon
                experienceMemorySizeSummary.value[0].simple_value = experienceMemory.size()
                writer.add_summary(explorationSummary, step)
                writer.add_summary(experienceMemorySizeSummary, step)
            # save checkpoint
            if step % CHECKPOINTS_STEPS == 0:
                saveModel(checkpointDirectory, step, sess)
            # calculate epsilon for next step
            epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) / (EPSILON_DECAY_STEPS / step)

        # write episodic summary
        episodicStepsSummary.value[0].simple_value = episodeSteps
        episodicRewardSummary.value[0].simple_value = episodeReward
        writer.add_summary(episodicStepsSummary, step)
        writer.add_summary(episodicRewardSummary, step)

if __name__ == "__main__":
    # execute only if run as a script
    main()
