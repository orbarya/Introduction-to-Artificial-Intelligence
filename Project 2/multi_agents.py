from math import inf

import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        bonus = 1
        if action == Action.LEFT or action == Action.DOWN:
            bonus = 4
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score

        "*** YOUR CODE HERE ***"
        empty_tiles = successor_game_state.get_empty_tiles()[0].shape[0]
        return bonus * score * max_tile * empty_tiles


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        legal_actions = game_state.get_legal_actions(0)
        v = -inf
        maximal_action = None
        for action in legal_actions:
            successor_value = self.min_value(game_state.generate_successor(0, action), 1)
            if successor_value > v:
                v = successor_value
                maximal_action = action
        return maximal_action

    def max_value(self, game_state, depth):
        depth = depth + 1

        print('Visiting max_value with depth ' + str(depth))
        print('The maximum depth allowed is ' + str(self.depth))

        # TODO  1. Does the max_value always checks the agents moves and the min_value always checks for the opponents moves?
        legal_actions = game_state.get_legal_actions(0)

        # TODO  2. Do we test for terminal state by checking if legal_actions is an empty list?
        if not legal_actions or depth > self.depth:
            return self.evaluation_function(game_state)
        v = -inf

        successors = self.get_successors(0, game_state, legal_actions)
        for successor in successors:
            v = max(v, self.min_value(successor, depth))
        return v

    def min_value(self, game_state, depth):

        # TODO  1. Does the max_value always checks the agents moves and the min_value always checks for the opponents moves?
        legal_actions = game_state.get_legal_actions(1)

        # TODO  2. Do we test for terminal state by checking if legal_actions is an empty list?
        if not legal_actions:
            return self.evaluation_function(game_state)
        v = inf
        successors = self.get_successors(1, game_state, legal_actions)
        for successor in successors:
            v = min(v, self.max_value(successor, depth))
        return v

    def get_successors(self, agent_index, game_state, legal_actions):
        successors = []
        for legal_action in legal_actions:
            successors.append(game_state.generate_successor(agent_index, legal_action))
        return successors


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()





def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
