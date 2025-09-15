# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qValues = util.Counter()

# ---------------------Q3---------------------#
    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        return max(self.getQValue(state, a) for a in actions)

    def computeActionFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None
        maxQ = self.computeValueFromQValues(state)
        best = [a for a in actions if self.getQValue(state, a) == maxQ]
        return random.choice(best)

    def getAction(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(actions)
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
#---------------------Q3---------------------#
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

# ---------------------Q4---------------------#
    def getAction(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None
        if util.flipCoin(self.epsilon):
            action = random.choice(actions)
        else:
            action = self.computeActionFromQValues(state)
        # --------------Q5-----------#
        self.doAction(state, action)
        # --------------Q5-----------#
        return action
#---------------------Q4---------------------#
class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

# ---------------------Q6---------------------#
    def getQValue(self, state, action):
        """
        Approximate Q(s,a) = sum_f w_f * f_f(s,a)
        """
        features = self.featExtractor.getFeatures(state, action)
        q = 0.0
        for f, val in features.items():
            q += self.weights[f] * val
        return q

    def update(self, state, action, nextState, reward):
        """
        w_f <- w_f + alpha * difference * f_f(s,a)
        where difference = [r + gamma * max_a' Q(nextState, a')] - Q(state, action)
        """
        # temporal-difference target - current estimate
        next_v = self.computeValueFromQValues(nextState)  # uses self.getQValue under the hood
        diff = (reward + self.discount * next_v) - self.getQValue(state, action)

        features = self.featExtractor.getFeatures(state, action)
        for f, val in features.items():
            self.weights[f] += self.alpha * diff * val
# ---------------------Q6---------------------#
    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
