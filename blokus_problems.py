import functools

import numpy as np

from board import Board
from search import SearchProblem, ucs
from search import a_star_search
import util


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)



#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        return state.state[0, 0] >= 0 and \
               state.state[self.board.board_h - 1, 0] >= 0 and \
               state.state[0, self.board.board_w - 1] >= 0 and \
               state.state[self.board.board_h - 1,self.board.board_w - 1] >= 0
        # util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        print('hello')
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        board = Board(self.board.board_w, self.board.board_h, 1, self.board.piece_list)
        empty_board_tile_num = np.abs(np.sum(board.state))
        for action in actions:
            board.add_move(0,action)
        actions_costs = empty_board_tile_num - np.abs(np.sum(board.state))
        return actions_costs

        # util.raiseNotDefined()


def is_point_area_empty(state, radius, i, j):
    from_i = max(0, i - radius)
    from_j = max(0, j - radius)
    to_i = min(state.board_h, i + radius)
    to_j = min(state.board_w, j + radius)
    return not np.any(state.state[from_i:to_i, from_j:to_j] + 1)


def bad_point(state, i, j):
    if state.state[i,j] == 0:
        return False
    elif (i - 1) >= 0 and state.state[i - 1, j] == 0:
        return True
    elif (j + 1) <= state.board_w - 1 and state.state[i, j + 1] == 0:
        return True
    elif (j - 1) >= 0 and state.state[i, j - 1] == 0:
        return True
    elif (i + 1) <= state.board_h - 1 and state.state[i + 1, j] == 0:
        return True
    return False


def bad_state(state, points):
    for (i,j) in points:
        if bad_point(state, i, j):
            return True
    return False

# def distanse_to_points(state, points):
#     distance_from_corners = 0
#     max_radius = max(state.board_w, state.board_h)
#     radius = 0
#     for (i,j) in points:
#         while(is_point_area_empty(state, radius, i, j) and radius < max_radius):
#             radius += 1
#         distance_from_corners += radius
#         radius = 0
#     return distance_from_corners

def distanse_to_points(state, points):
    distance_from_corners = 0
    max_radius = max(state.board_w, state.board_h)
    radius = 0
    for (i,j) in points:
        while(is_point_area_empty(state, radius, i, j) and radius < max_radius):
            radius += 1
        if distance_from_corners <= radius:
            distance_from_corners = radius
        radius = 0
    if bad_state(state, points):
        return state.board_w * state.board_h
    return distance_from_corners


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    return distanse_to_points(state, [(state.board_h-1, state.board_w-1), (state.board_h-1, 0), (0, state.board_w-1), (0,0)])


def flip_targets (targets):
    return list(map(lambda target: (target[1], target[0]), targets))


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = flip_targets(targets)
        self.expanded = 0
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        return functools.reduce((lambda is_goal, target: is_goal and (state.get_position(target[1],
                                target[0])) != -1), self.targets,
                                True)

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        board = Board(self.board.board_w, self.board.board_h, 1, self.board.piece_list)
        empty_board_tile_num = np.abs(np.sum(board.state))
        for action in actions:
            board.add_move(0,action)
        actions_costs = empty_board_tile_num - np.abs(np.sum(board.state))
        return actions_costs


def blokus_cover_heuristic(state, problem):
    return distanse_to_points(state, problem.targets)


class BlokusExistingBoardCoverProblem(SearchProblem):
    def __init__(self, board, targets):
        self.targets = targets.copy()
        self.expanded = 0
        self.board = board.__copy__()

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        return functools.reduce((lambda is_goal, target: is_goal and (state.get_position(target[1],
                                target[0])) != -1), self.targets,
                                True)

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        board = Board(self.board.board_w, self.board.board_h, 1, self.board.piece_list)
        empty_board_tile_num = np.abs(np.sum(board.state))
        for action in actions:
            board.add_move(0, action)
        actions_costs = empty_board_tile_num - np.abs(np.sum(board.state))
        return actions_costs


class ClosestLocationSearch:


    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = flip_targets(targets)
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.piece_list = piece_list
        self.starting_point = starting_point

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        current_state = self.board.__copy__()
        backtrace = []
        remaining_targets = self.targets.copy()
        while len(remaining_targets) > 0:
            target = self.find_closest_target(current_state, remaining_targets)
            actions = self.cover_closest_target(current_state, target)
            for action in actions:
                current_state.add_move(0, action)
            backtrace.extend(actions)
            remaining_targets.remove(target)
        return backtrace

    def find_closest_target(self, current_state, remaining_targets):
        """

        :param current_state:
        :param remaining_targets:
        :return:
        """
        min_distance = np.inf
        closest_target = None
        for target in remaining_targets:
            d = distanse_to_points(current_state, [target])
            if d < min_distance:
                closest_target = target
                min_distance = d
        return closest_target

    def cover_closest_target(self, current_state, target):
        """

        :param current_state:
        :param target:
        :return:
        """
        target_cover_problem = BlokusExistingBoardCoverProblem(current_state, [target])
        actions = a_star_search(target_cover_problem, blokus_cover_heuristic)
        self.expanded += target_cover_problem.expanded
        return actions


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.blokusCoverProblem = BlokusCoverProblem(board_w, board_h, piece_list, starting_point, targets)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.blokusCoverProblem.get_start_state()

    def solve(self):
        return a_star_search(self.blokusCoverProblem, blokus_cover_heuristic)
