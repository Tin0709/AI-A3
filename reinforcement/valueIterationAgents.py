# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

#-------------------- Question 1 ------------------#
    def runValueIteration(self):
        """
        Run K steps of *batch* value iteration.
        Batch = compute V_{k+1} from a frozen copy of V_k for all states,
        then replace self values at the end of the sweep.
        """
        for _ in range(self.iterations):
            newValues = util.Counter()  # defaults to 0
            for s in self.mdp.getStates():
                # Terminal or no actions -> value is 0 by convention
                if self.mdp.isTerminal(s):
                    newValues[s] = 0.0
                    continue
                actions = self.mdp.getPossibleActions(s)
                if not actions:
                    newValues[s] = 0.0
                    continue
                # Bellman optimality backup using *old* self.values
                q_candidates = [self.computeQValueFromValues(s, a) for a in actions]
                newValues[s] = max(q_candidates)
            # commit the batch
            self.values = newValues

    def computeQValueFromValues(self, state, action):
        """
        Q(s,a) = sum_{s'} P(s'|s,a) * [ R(s,a,s') + gamma * V(s') ]
        Uses current self.values (i.e., V_k).
        """
        q = 0.0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            q += prob * (reward + self.discount * self.values[nextState])
        return q

    def computeActionFromValues(self, state):
        """
        Return argmax_a Q(s,a) under current self.values.
        If terminal or no legal actions, return None.
        """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        # compute best action by Q-value
        best_action, best_val = None, float("-inf")
        for a in actions:
            q = self.computeQValueFromValues(state, a)
            if q > best_val:
                best_val, best_action = q, a
        return best_action
    # -------------------- Question 1 ------------------#

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
