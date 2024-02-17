# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # import partial functionality for easy function shortcutting and reuse
        from functools import partial
        score = 0
        # derive list from exisiting food
        remainingFood = newFood.asList()
        # create a partial of the manhattanDistance function, taking newPos as input
        manPart = partial(manhattanDistance, newPos)
        
        # create a list comprehension of the manhattan partial applied to each ghost state's distance
        ghostDistances = [manPart(ghost.getPosition()) for ghost in newGhostStates]
        # if a move results in colliding with a ghost and thus dying, this is the worst possible move
        for item in ghostDistances:
            if item <= 1:
                return -9999999
        # add the closest ghost's distance to the score; the farther away, the better
        score += min(ghostDistances)
        
        # like ghosts, create a list comprehension of the manhattan partial applied to the position of the remaining food pellets
        foodDistances = [manPart(pellet) for pellet in remainingFood]
        # reduce food distance to 0 if none remains, otherwise the closest pellet is chosen
        closestFoodDistance = 0 if len(remainingFood) == 0 else min(foodDistances)
        # dramatically disincentivize passing on a move, slightly reward moving 
        score -= 750 if action == 'STOP' else -5
        # incentivize moves that consume pellets
        score += 30 if currentGameState.getNumFood() > successorGameState.getNumFood() else 0
        # add the base successor state score and the negative of the closest food pellet to the score
        # the farther the closest food pellet, the worse the score
        score += -closestFoodDistance + successorGameState.getScore()
        #print(score)
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # initialize the values representing the best possible score and action
        optimalScore = -9999999
        optimalAction = None
        """ 
        iterate through all legal actions for Pacman, 
        update the optimal score and its associated action 
        when a better one arises. use minimax to evaluate 
        scores based on the possible states of both Pacman
        and the ghosts up to a certain depth 
        """
        for action in gameState.getLegalActions(0): # index 0 == Pacman
            successor = gameState.generateSuccessor(0, action)
            score = self.minimax(successor, self.depth, 1)
            if score > optimalScore:
                optimalScore = score
                optimalAction = action
        return optimalAction
        # util.raiseNotDefined()
    
    # minimax function, takes turns with Pacman (max, index == 0) and ghosts (min, index > 0)
    # also returns terminal states when relevant
    def minimax(self, state, depth, index):
        return (self.evaluationFunction(state) if state.isWin() or state.isLose() or depth == 0
                else self.minifyMaxify(state, depth, index, False) if index == 0
                else self.minifyMaxify(state, depth, index, True))

    # finds maximum or minimum score legal actions for Pacman or a ghost, respectively
    def minifyMaxify(self, state, depth, index, isGhost):
        # initialize score for respective agent
        score = -9999999 if not isGhost else 9999999
        # find all legal actions the agent can take this turn
        actions = state.getLegalActions(index)
        # return terminal if no legal actions
        if not actions: return self.evaluationFunction(state)
        # iterate through legal actions and find best respective (min or max) score for agent
        for action in actions:
            # get next successor state
            s = state.generateSuccessor(index, action)
            """
            Nested ternary operator expressions:
            
            "if the current agent's a ghost, find its minimal score and then either
            iterate to Pacman (index 0) or the next ghost (index+1) depending on whether
            the current agent is the last ghost or not. otherwise, if the agent is not 
            a ghost (and thus Pacman), find its maximum score and iterate to the first 
            ghost (index 1)."

            the first and second lines represent the internal ternary operator, which 
            itself is the first conditional value of the ternary operator reliant on
            said condition being true

            the third line represents the the external ternary operator's second
            conditional value, and runs for all other cases than when the first value's
            condition is true
            """
            score = ((min(score, self.minimax(s, depth-1, 0)) if index == state.getNumAgents()-1
                        else min(score, self.minimax(s, depth, index+1))) if isGhost
                    else max(score, self.minimax(s, depth, 1)))
        return score
            


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    
    """ 
    getAction remains virtually the same as Q2 other than some minor syntax 
    changes made to increase concision and reduce type and scope errors
    """
    def getAction(self, gameState: GameState):
        # initialize the values representing the best possible score and action
        optimalScore = -9999999
        optimalAction = None
        """ 
        iterate through all legal actions for Pacman, 
        update the optimal score and its associated action 
        when a better one arises. use expectimax to evaluate 
        scores based on the possible states of both Pacman
        and the ghosts up to a certain depth 
        """
        for action in gameState.getLegalActions(0): # index 0 == Pacman
            successor = gameState.generateSuccessor(0, action)
            score = self.expectimax(successor, self.depth, 1)[0] # reduce scope to score, minimizing expanded state errors
            if score > optimalScore:
                optimalScore, optimalAction = score, action # shortcut syntax
        return optimalAction
        # util.raiseNotDefined()
    
    """ 
    expectimax function forked from minimax, remaining mostly the same except for its ternary operator
    being reduced to two conditional values due to changes in expectifyMaxify
    
    the terminal state was also adjusted to maintain consistency with the action tuple return type
    """
    def expectimax(self, state, depth, index):
        return ((self.evaluationFunction(state), None) if state.isWin() or state.isLose() or depth == 0
                else self.expectifyMaxify(state, depth, index))

    """ 
    expectifyMaxify is an extensively reworked minifyMaxify fork, ditching the isGhost logic
    in favor of further homogenizing the logic between Pacman and the ghosts
    
    scores are now tracked as a list and evaluated respective to agent type after the for loop, 
    and the nested ternary operator was swapped out for a modulo index variable and ternary
    operator depth variable. scores themselves are appended to the score list and all agents
    go through expectimax in-loop
    
    once the loop is finished, Pacman is assigned the same max score and ghosts are assigned an
    average of the scores a la expectimax
    """
    # finds maximum or expected score out of all legal actions for Pacman or a ghost, respectively
    def expectifyMaxify(self, state, depth, index):
        # find all legal actions the agent can take this turn
        actions = state.getLegalActions(index)
        # return terminal with action tuple return type if no legal actions
        if not actions: return self.evaluationFunction(state), None
        
        # iterate through legal actions and find best respective (exp or max) score for agent
        scores = []
        for action in actions:
            # get next successor state
            s = state.generateSuccessor(index, action)
            nextIndex = (index + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextIndex == 0 else depth  # decrease depth on full cycle through agents
            # call expectimax recursively for next agent
            score = self.expectimax(s, nextDepth, nextIndex)[0]
            scores.append(score)
        # optimize score for respective agent type (Pacman at index 0 or ghosts otherwise)
        bestScore = max(scores) if index == 0 else sum(scores)/len(scores)
        return bestScore, None  # return with action tuple return type for consistency

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    This function is a fundamental redesign of the evaluationFunction in Q1 with a better understanding of
    and emphasis on using incentives to encourage or discourage certain behaviors. The function calculates
    a variety of positive and negative incentives that impact the final score of any given move. 
    
    Being far away from food is negatively incentivized, discouraging Pacman from staying in spaces where no
    food exists.
    
    Ghosts inherently apply a negative incentive. The weight of this incentive is increased by distance from
    scared ghosts, encouraging aggression against them. It is also decreased by distance from active ghosts,
    encouraging defensive behavior.
    
    Capsules apply a negative incentive that is gradually reduced to 0 as they're consumed, encouraging Pacman
    to take advantage of them as early as possible to get ahead.
    
    The number of food pellets remaining grants a ramping positive incentive as it decreases, encouraging Pacman
    to prioritize consuming food pellets.
    
    Finally, reducing the remaining food pellets to less than five applies a massive positive incentive, encouraging
    Pacman to finish strong with a reward.
    
    This function improves on the old evaluation function massively by reworking existing incentives and introducing
    incentives all with a careful central balance, ensuring that Pacman more advantageously prioritizes old behaviors
    while taking into account new behaviors that are both beneficial and harmful to him.
    """
    # immediately return incentivized score if game ending states are reached
    if currentGameState.isWin(): return 9999999 # max positive incentive for winning states
    elif currentGameState.isLose(): return -9999999 # max negative incentive for losing states
    
    # initialize variables
    score = currentGameState.getScore() # initialize score
    pacmanPosition = currentGameState.getPacmanPosition() # initialize pacman position
    foodList = currentGameState.getFood().asList() # get list of remaining food
    remainingFood = len(foodList) # quantify remaining food
    foodDistanceIncentive = 0 # create positive incentive based on distance from nearest food pellet
    ghostIncentive = 0 # create negative incentive based on distance from nearest ghost
    ghostStates = currentGameState.getGhostStates() # get states of all ghosts
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates] # get times that all ghosts will be scared for
    
    # adjust food incentive using manhattan distance
    foodDistances = [manhattanDistance(pacmanPosition, food) for food in foodList] # find all manhattan distances in foodList with list comprehension
    nearestFoodDistance = min(foodDistances) # find nearest food pellet
    foodDistanceIncentive = 50 / nearestFoodDistance  # divide a positive incentive by food distance; the closer the better

    # iterate through ghost states and determine incentives for scared and active ghosts
    for i, ghost in enumerate(ghostStates):
        distance = manhattanDistance(pacmanPosition, ghost.getPosition())
        if scaredTimes[i] > 0:
            # increase negative ghost incentive by a negative incentive divided by distance; the closer the better
            if distance > 0: ghostIncentive += 200 / distance # encourage aggression against scared ghosts
        # decrease negative ghost incentive by a positive incentive divided by distance; the farther the better
        elif distance <= 2: ghostIncentive -= 500 / (distance + 0.1) # discourages aggression against active ghosts

    # append some miscellaneous incentives
    capsuleIncentive = -30 * len(currentGameState.getCapsules()) # negative incentive for leaving capsules uneaten
    remainingFoodIncentive = 150 * (1 / (1 + remainingFood)) # positive incentive for decreasing remaining food
    # append special incentive to encourage aggressive late-game strategy
    if remainingFood < 5: score += 500 # positive incentive to finish up last few pellets

    # add up all incentives into a final score and return it
    score += foodDistanceIncentive + ghostIncentive + capsuleIncentive + remainingFoodIncentive
    return score

# Abbreviation
better = betterEvaluationFunction
