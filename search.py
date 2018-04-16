"""
In search.py, you will implement generic search algorithms
"""
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def visit_node(problem, parents, visited, to_visit, current_state):
    descendants = problem.get_successors(current_state)
    # descendants.reverse()
    for descendant in descendants:
        if descendant[0] not in parents:
            parents[descendant[0]] = (current_state, descendant[1], descendant[2])
    to_visit += descendants
    visited.add(current_state)
    return parents, visited, to_visit


def get_actions_list(parents, current_state, start):
    actions = []
    while current_state is not start:
        actions.append(parents[current_state][1])
        current_state = parents[current_state][0]
    actions.reverse()   
    return actions

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    start = problem.get_start_state()
    current_state = start 
    parents = {}
    visited = set()
    to_visit = []
    parents, visited, to_visit = visit_node(problem, parents, visited, to_visit, current_state)
    
    while(len(to_visit) > 0 and not problem.is_goal_state(current_state)):
        current_state = to_visit.pop()[0]
        if current_state not in visited:
            parents, visited, new_nodes = visit_node(problem, parents, visited, to_visit, current_state)
    to_visit += new_nodes

    # print(problem.is_goal_state(current_state))
    return get_actions_list(parents, current_state, start)
    # util.raiseNotDefined()


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    start = problem.get_start_state()
    current_state = start 
    parents = {current_state: 'start'}
    visited = set()
    to_visit = []
    parents, visited, new_nodes = visit_node(problem, parents, visited, to_visit, current_state)
    to_visit += new_nodes
    
    while not problem.is_goal_state(current_state):
        if len(to_visit) > 0:
            return []
        current_state = to_visit.pop(0)[0]
        if current_state not in visited:
            parents, visited, to_visit = visit_node(problem, parents, visited, to_visit, current_state)
    
    # print(problem.is_goal_state(current_state))
    return get_actions_list(parents, current_state, start)
    # util.raiseNotDefined()


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    return best_first_search(problem, null_heuristic)


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    return best_first_search(problem, heuristic)


def best_first_search(problem, heuristic=null_heuristic):
    initial_state = problem.get_start_state()
    explored_set = set()
    in_frontier = set()
    came_from = {}
    g_score = {}
    g_score[initial_state] = 0
    f_score = {}
    g_score[initial_state] = heuristic(initial_state, problem)

    frontier = util.PriorityQueueWithFunction(lambda item: g_score[item] + heuristic(item, problem))
    frontier.push(initial_state)
    in_frontier.add(initial_state)

    while not frontier.isEmpty():
        current_state = frontier.pop()
        in_frontier.remove(current_state)

        if current_state in explored_set:
            continue

        if problem.is_goal_state(current_state):
            return get_actions_list(came_from, current_state, initial_state)
        explored_set.add(current_state)

        neighbors = problem.get_successors(current_state)
        for neighbor in neighbors:
            if neighbor[0] in explored_set:
                continue
            tentative_g_score = g_score[current_state] + neighbor[2]
            if tentative_g_score >= g_score.get(neighbor[0], float("inf")):
                continue

            came_from[neighbor[0]] = [current_state, neighbor[1]]
            g_score[neighbor[0]] = tentative_g_score
            #if neighbor[0] not in frontier:
            frontier.push(neighbor[0])
            in_frontier.add(neighbor[0])
    return []


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
