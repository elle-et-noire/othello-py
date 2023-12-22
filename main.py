from enum import Enum
import copy
import random
import time

class Stone(Enum):
    EMPTY = 0
    BLACK = -1
    WHITE = 1
    WALL = 334

    def flip(self):
        if self.value == self.WALL:
            print("wall cannot be flipped.")
        return Stone(self.value * -1)
    
    @staticmethod
    def values():
        return [Stone.EMPTY, Stone.BLACK, Stone.WHITE, Stone.WALL]
    
    def ordinal(self):
        if self.value == self.EMPTY.value:
            return 0
        if self.value == self.BLACK.value:
            return 1
        if self.value == self.WHITE.value:
            return 2
        return 3

class Position():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equals(self, p):
        return self.x == p.x and self.y == p.y
    
    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

class Board():
    WIDTH = 8
    HEIGHT = 8
    MAP_WIDTH = WIDTH + 2
    MAP_HEIGHT = HEIGHT + 2
    DX = [1, 1, 0, -1, -1, -1, 0, 1]
    DY = [0, 1, 1, 1, 0, -1, -1, -1]

    def __init__(self):
        self.board = [[Stone.EMPTY for i in range(self.MAP_WIDTH)] for j in range(self.MAP_HEIGHT)]
        for i in range(self.MAP_HEIGHT):
            self.board[i][0] = self.board[i][self.MAP_WIDTH - 1] = Stone.WALL
        for i in range(self.MAP_WIDTH):
            self.board[0][i] = self.board[self.MAP_HEIGHT - 1][i] = Stone.WALL

    def get(self, x, y):
        return self.board[y][x]

    def set(self, x, y, stone):
        self.board[y][x] = stone

    def setup(self):
        for y in range(1, self.HEIGHT + 1):
            for x in range(1, self.WIDTH + 1):
                self.set(x, y, Stone.EMPTY)
        self.set(4, 4, Stone.WHITE)
        self.set(5, 5, Stone.WHITE)
        self.set(4, 5, Stone.BLACK)
        self.set(5, 4, Stone.BLACK)

    #isPuttableで確認後呼び出す
    def put(self, x, y, stone):
        self.set(x, y, stone)
        for i in range(8):
            count = self.countFlippable(x, y, stone, self.DX[i], self.DY[i])
            for j in range(1, count + 1):
                self.set(x + self.DX[i] * j, y + self.DY[i] * j, stone)

    def countFlippable(self, x, y, stone, dx, dy):
        count = 0
        yy = y + dy
        xx = x + dx
        opponent = stone.flip()
        while self.get(xx, yy) == opponent:
            count += 1
            yy += dy
            xx += dx

        if self.get(xx, yy) == stone:
            return count
        #進んだ先がEMPTYかWALL
        return 0

    def isPuttable(self, x, y, stone):
        if not (1 <= x <= self.WIDTH and 1 <= y <= self.HEIGHT):
            return False
        if self.get(x, y) != Stone.EMPTY:
            return False

        for i in range(8):
            if self.countFlippable(x, y, stone, self.DX[i], self.DY[i]) > 0:
                return True
        return False

    def isPuttableSomewhere(self, stone):
        for y in range(1, self.HEIGHT + 1):
            for x in range(1, self.WIDTH + 1):
                if self.isPuttable(x, y, stone):
                    return True
        return False

    def countStone(self, stone):
        count = 0
        for y in range(1, self.HEIGHT + 1):
            for x in range(1, self.WIDTH + 1):
                if self.get(x, y) == stone:
                    count += 1
        return count

    def findPuttableHands(self, stone):
        ps = []
        for y in range(1, self.HEIGHT + 1):
            for x in range(1, self.WIDTH + 1):
                if self.isPuttable(x, y, stone):
                    ps.append(Position(x, y))
        return ps

    def clone(self):
        b = Board()
        b.board = copy.deepcopy(self.board)
        return b

    def str(self):
        s = ' '
        for x in range(1, self.WIDTH + 1):
            s += str(x)
        s += '\n'
        for y in range(1, self.HEIGHT + 1):
            s += str(y)
            for x in range(1, self.WIDTH + 1):
                if self.get(x, y) == Stone.EMPTY:
                    s += '.'
                if self.get(x, y) == Stone.BLACK:
                    s += 'B'
                if self.get(x, y) == Stone.WHITE:
                    s += 'w'
                if self.get(x, y) == Stone.WALL:
                    s += '*'
            s += '\n'

        return s

    def show(self):
        print(self.str())

    def getHorizontal(self, y):
        return [self.get(i + 1, y) for i in range(self.WIDTH)]
    
    def getVertical(self, x):
        return [self.get(x, i + 1) for i in range(self.HEIGHT)]

class Turn(Enum):
    BLACK = -1
    WHITE = 1

    def flip(self):
        return Turn(self.value * -1)

    def stone(self):
        #先手は黒、後手は白
        return Stone(self.value)
    
class Player():
    def __init__(self, turn):
        self.turn = turn

    def getTurn(self):
        return self.turn

class HumanPlayer(Player):
    def __init__(self, turn):
        self.turn = turn

    def play(self, board):
        while True:
            print('(x, y) = ', end = '')
            x, y = (int(x) for x in input().split())
            if not board.isPuttable(x, y, self.turn.stone()):
                print('Invalid.')
                continue

            return Position(x, y)

class AbstractAIPlayer(Player):
    def __init__(self, turn, maxDepth):
        self.turn = turn
        self.maxDepth = maxDepth
        
    def play(self, board):
        return self.makeSearcher().eval(board, self.maxDepth, self.turn,
                    self.makeEvaluator()).getPosition()

class BoardScoreEvaluator():
    EVAL_VALUES = [
        [100, -50, 35, 30, 30, 35, -50, 100],
        [-50, -70, 10, 15, 15, 10, -70, -50],
        [35, 10, 20, 25, 25, 20, 10, 35],
        [30, 15, 25, 50, 50, 25, 15, 30],
        [30, 15, 25, 50, 50, 25, 15, 30],
        [35, 10, 20, 25, 25, 20, 10, 35],
        [-50, -70, 10, 15, 15, 10, -70, -50],
        [100, -50, 35, 30, 30, 35, -50, 100]
    ]
    
    def __init__(self, turn, score=0):
        self.turn = turn
        self.currentScore = score

    def clone(self):
        return BoardScoreEvaluator(self.turn, self.currentScore)
    
    def willPut(self, board, x, y, stone):
        if stone == self.turn.stone():
            self.currentScore += self.EVAL_VALUES[y - 1][x - 1]
        else:
            self.currentScore -= self.EVAL_VALUES[y - 1][x - 1]

    def score(self, board, currentTurn):
        if currentTurn == self.turn:
            return self.currentScore
        else:
            return -self.currentScore
        
class NegaMaxSearcher():
    def eval(self, board, restDepth, currentTurn, evaluator):
        if restDepth == 0:
            return EvalResult(evaluator.score(board, currentTurn), None)
        
        puttablePositions = board.findPuttableHands(currentTurn.stone())
        if len(puttablePositions) == 0:
            score = -self.eval(board, restDepth - 1, currentTurn.flip(), evaluator).getScore()
            return EvalResult(score, None)
        
        maxScore = -float('inf')
        selectedPosition = None

        for p in puttablePositions:
            b = board.clone()
            e = evaluator.clone()
            
            e.willPut(b, p.x, p.y, currentTurn.stone())
            b.put(p.x, p.y, currentTurn.stone())

            score = -self.eval(b, restDepth - 1, currentTurn.flip(), e).getScore()
            if maxScore < score:
                maxScore = score
                selectedPosition = p

        return EvalResult(maxScore, selectedPosition)
    
class NegaScoutBoardScoreAIPlayer(AbstractAIPlayer):
    def __init__(self, turn, maxDepth):
        super().__init__(turn, maxDepth)

    def makeEvaluator(self):
        return BoardScoreEvaluator(self.turn)
    
    def makeSearcher(self):
        return NegaScoutSearcher()

class SimpleAIPlayer():
    def __init__(self, turn):
        self.turn = turn

    def play(self, board):
        pos = Position(0, 0)
        maxVal = -float('inf')

        for p in board.findPuttableHands(self.turn.stone()):
            if maxVal < BoardScoreEvaluator.EVAL_VALUES[p.y - 1][p.x - 1]:
                maxVal = BoardScoreEvaluator.EVAL_VALUES[p.y - 1][p.x - 1]
                pos = p

        return pos

class EvalResult():
    def __init__(self, score, position):
        self.score = score
        self.position = position

    def getScore(self):
        return self.score
    
    def getPosition(self):
        return self.position

class FixedStones:
    @staticmethod
    def fromInt(idx):
        board = Board()
        for x in range(8, 0, -1):
            board.set(x, 1, Stone.values()[idx % 3])
            idx //= 3

        return board

    @staticmethod
    def toInt(stones):
        result = 0
        for s in stones:
            result = result * 3 + s.ordinal()

        return result

    @staticmethod
    def calculateIter(ith, visited):
        if visited[ith] >= 0:
            return visited[ith]

        board = FixedStones.fromInt(ith)
        foundEmpty = False
        result = 100

        for x in range(1, Board.WIDTH+1):
            if board.get(x, 1) != Stone.EMPTY:
                continue

            foundEmpty = True

            b = board.clone()
            b.put(x, 1, Stone.BLACK)
            result = min(result, FixedStones.calculateIter(FixedStones.toInt(b.getHorizontal(1)), visited))

            b = board.clone()
            b.put(x, 1, Stone.WHITE)
            result = min(result, FixedStones.calculateIter(FixedStones.toInt(b.getHorizontal(1)), visited))

        if foundEmpty:
            visited[ith] = result
            return result

        numBlackStone = 0
        for i in range(1, Board.WIDTH+1):
            if board.get(i, 1) == Stone.BLACK:
                numBlackStone += 1

        visited[ith] = numBlackStone
        return numBlackStone
    
    @staticmethod
    def calculate():
        result = [0 for _ in range(6561)]
        for i in range(6561):
            result[i] = FixedStones.calculateIter(i, result)
        return result

    fixedStone = []
    initialized = False

    @staticmethod
    def init():
        if not FixedStones.initialized:
            FixedStones.fixedStone = FixedStones.calculate()
            FixedStones.initialized = True

    @staticmethod
    def getNumFixedStones(board, stone):
        upper = board.getHorizontal(1)
        lower = board.getHorizontal(Board.HEIGHT)
        left = board.getVertical(1)
        right = board.getVertical(Board.WIDTH)

        if stone != Stone.BLACK:
            for i in range(8):
                upper[i] = upper[i].flip()
                lower[i] = lower[i].flip()
                left[i] = left[i].flip()
                right[i] = right[i].flip()

        count = 0
        count += FixedStones.fixedStone[FixedStones.toInt(upper)]

        if board.get(1, 1) == stone:
            count -= 1
        if board.get(1, Board.HEIGHT) == stone:
            count -= 1
        if board.get(Board.WIDTH, 1) == stone:
            count -= 1
        if board.get(Board.WIDTH, Board.HEIGHT) == stone:
            count -= 1

        return count

class AdvancedEvaluator():
    def __init__(self, turn, currentOpenness = 0):
        self.turn = turn
        self.currentOpenness = currentOpenness
        FixedStones.init()

    def clone(self):
        return AdvancedEvaluator(self.turn, self.currentOpenness)

    def willPut(self, board, x, y, stone):
        for i in range(len(Board.DX)):
            dx = Board.DX[i]
            dy = Board.DY[i]
            count = board.countFlippable(x, y, stone, dx, dy)
            # 開放度を計算して足し算
            for j in range(1, count+1):
                if self.turn.stone() == stone:
                    self.currentOpenness += self.countOpenness(board, x + dx * j, y + dy * j)
                else:
                    self.currentOpenness -= self.countOpenness(board, x + dx * j, y + dy * j)

    def countOpenness(self, board, x, y):
        count = 0
        for i in range(len(Board.DX)):
            dx = Board.DX[i]
            dy = Board.DY[i]
            if board.get(x + dx, y + dy) == Stone.EMPTY:
                count += 1
        return count

    def score(self, board, currentTurn):
        # 開放度を加点
        v = -self.currentOpenness * 3
        # 自分の確定石を加点
        v += FixedStones.getNumFixedStones(board, self.turn.stone()) * 20
        # 相手の確定石を減点
        v -= FixedStones.getNumFixedStones(board, self.turn.flip().stone()) * 20
        # 置ける箇所を加点
        v += len(board.findPuttableHands(self.turn.stone()))
        # 相手の置ける箇所を減点
        v -= len(board.findPuttableHands(self.turn.flip().stone()))

        return v if currentTurn == self.turn else -v

class NegaMaxAdvancedAIPlayer(AbstractAIPlayer):
    def __init__(self, turn, maxDepth):
        super().__init__(turn, maxDepth)

    def makeEvaluator(self):
        return AdvancedEvaluator(self.turn)
    
    def makeSearcher(self):
        return NegaMaxSearcher()
    
class AlphaBetaSearcher():
    def eval(self, board, restDepth, currentTurn, scoreSum, alpha, beta):
        if restDepth == 0:
            return EvalResult(scoreSum if currentTurn == self.turn else -scoreSum, None)

        puttables = board.findPuttableHands(currentTurn.stone())

        if len(puttables) == 0:
            score = -self.eval(board, restDepth - 1, currentTurn.flip(), scoreSum, -beta, -alpha).score
            return EvalResult(score, None)

        maxScore = -float('inf')
        selected = None

        for p in puttables:
            b = board.clone()
            b.put(p.x, p.y, currentTurn.stone())
            scoreDiff = (1 if currentTurn == self.turn else -1) * self.EVAL_VALUES[p.y - 1][p.x - 1]
            score = -self.eval(b, restDepth - 1, currentTurn.flip(), scoreSum + scoreDiff, -beta, -max(alpha, maxScore)).score
            if maxScore < score:
                maxScore = score
                selected = p

            if maxScore >= beta:
                return EvalResult(score, p)

        return EvalResult(maxScore, selected)
    
class NegaScoutSearcher():
    def eval(self, board, restDepth, currentTurn, evaluator, alpha=-float('inf'), beta=float('inf')):
        if restDepth == 0:
            return EvalResult(evaluator.score(board, currentTurn), None)
        
        puttablePositions = board.findPuttableHands(currentTurn.stone())

        if len(puttablePositions) == 0:
            print("uo")
            score = -self.eval(board, restDepth - 1, currentTurn.flip(), evaluator, -beta, -alpha).getScore()
            return EvalResult(score, 0)
        
        maxScore = -float('inf')
        selectedPosition = 1.1

        for p in puttablePositions:
            b = board.clone()
            e = evaluator.clone()

            e.willPut(b, p.x, p.y, currentTurn.stone())
            b.put(p.x, p.y, currentTurn.stone())

            a = max(alpha, maxScore)
            score = -self.eval(b, restDepth - 1, currentTurn.flip(), e, -a - 1, -a).getScore()
            if (a < score < beta):
                e = evaluator.clone()
                e.willPut(board, p.x, p.y, currentTurn.stone())
                score = -self.eval(b, restDepth - 1, currentTurn.flip(), e, -beta, -score).getScore()

            if maxScore < score:
                maxScore = score
                selectedPosition = p
            
            # beta cut
            if maxScore >= beta:
                return EvalResult(score, p)
            
        return EvalResult(maxScore, selectedPosition)
    
class NegaScoutAIPlayer(AbstractAIPlayer):
    def __init__(self, turn, maxDepth):
        super().__init__(turn, maxDepth)

    def makeEvaluator(self):
        return AdvancedEvaluator(self.turn)
    
    def makeSearcher(self):
        return NegaScoutSearcher()
    
# class PerfectPlayWrapperPlayer(Player):
#     def __init__(self, originalPlayer, completeReadingThreshold):
#         self.turn = originalPlayer.getTurn()
#         self.originalPlayer = originalPlayer
#         self.completeReadingThreshold = completeReadingThreshold

#     def play(self, board):
#         restHand = 64 - board.countStones()
#         print(restHand)

#         if restHand <= self.completeReadingThreshold:
#             beginTime = time.time()
#             p = self.originalPlayer.makeSearcher().eval(board, restHand, self.turn, NumStoneEvaluator(self.turn)).getPosition()
#             endTime = time.time()
#             print("reading time = " + (endTime - beginTime) + "[s]")

#         return self.originalPlayer.play(board)
    
class Winner(Enum):
    BLACK = -1
    WHITE = 1
    TIE = 0

    def __str__(self):
        if self.value == self.BLACK.value:
            return "BLACK"
        if self.value == self.WHITE.value:
            return "WHITE"
        if self.value == self.TIE.value:
            return "TIE"

class Game():
    def play(self, blackPlayer, whitePlayer, board, turn, verbose):
        hasPassed = False

        while True:
            if verbose:
                board.show()
            
            if not board.isPuttableSomewhere(turn.stone()):
                if hasPassed:
                    break

                hasPassed = True
                turn = turn.flip()
                continue

            print(board.findPuttableHands(turn.stone())[0])
            hasPassed = False
            p = blackPlayer.play(board) if turn == Turn.BLACK else whitePlayer.play(board)

            board.put(p.x, p.y, turn.stone())
            turn = turn.flip()

        blackStoneCount = board.countStone(Stone.BLACK)
        whiteStoneCount = board.countStone(Stone.WHITE)

        if verbose:
            board.show()
            print("BLACK = " + str(blackStoneCount))
            print("WHITE = " + str(whiteStoneCount))

        if blackStoneCount > whiteStoneCount:
            return Winner.BLACK
        elif blackStoneCount < whiteStoneCount:
            return Winner.WHITE
        else:
            return Winner.TIE
        
class RandomPlayer(Player):
    def __init__(self, turn):
        self.turn = turn

    def play(self, board):
        while True:
            x = random.randrange(Board.WIDTH) + 1
            y = random.randrange(Board.HEIGHT) + 1
            if not board.isPuttable(x, y, self.turn.stone()):
                continue

            return Position(x, y)
        
class EvalWinRate():
    GAME_COUNT = 10
    
    @staticmethod
    def main():
        blackPlayer = SimpleMonteCarloPlayer(Turn.BLACK, 10)
        whitePlayer = NegaScoutAIPlayer(Turn.WHITE, 5)

        black = 0; white = 0; tie = 0

        game = Game()
        for i in range(EvalWinRate.GAME_COUNT):
            board = Board()
            board.setup()
            winner = game.play(blackPlayer, whitePlayer, board, Turn.BLACK, False)

            if winner == Winner.BLACK:
                black += 1
            elif winner == Winner.WHITE:
                white += 1
            elif winner == Winner.TIE:
                tie += 1
            else:
                print("Unknown winner.")

            print("winner: " + str(winner) + " (at " + str(i + 1) + ")")

        print("BLACK: " + type(blackPlayer).__name__)
        print("WHITE: " + type(whitePlayer).__name__)
        print("black: " + str(black / EvalWinRate.GAME_COUNT))
        print("white: " + str(white / EvalWinRate.GAME_COUNT))
        print("tie: " + str(tie / EvalWinRate.GAME_COUNT))

class SimpleMonteCarloPlayer(Player):
    def __init__(self, turn, playoutCount):
        self.turn = turn
        self.playoutCount = playoutCount

    def play(self, board):
        start = time.time()
        maxRate = -1
        maxPosition = None
        for y in range(1, Board.HEIGHT + 1):
            for x in range(1, Board.WIDTH + 1):
                if not board.isPuttable(x, y, self.turn.stone()):
                    continue
                rate = self.playout(x, y, board)
                if rate > maxRate:
                    maxRate = rate
                    maxPosition = Position(x, y)

        end = time.time()
        print("play duration: " + str(end - start))
        return maxPosition

    def playout(self, x, y, board):
        blackPlayer = RandomPlayer(Turn.BLACK)
        whitePlayer = RandomPlayer(Turn.WHITE)

        nextBoard = board.clone()
        nextBoard.put(x, y, self.turn.stone())

        win = 0
        game = Game()
        for count in range(0, self.playoutCount):
            winner = game.play(blackPlayer, whitePlayer, nextBoard.clone(), self.turn.flip(), False)
            if self.turn == Turn.BLACK and winner == Winner.BLACK or self.turn == Turn.WHITE and winner == Winner.WHITE:
                win += 1

        return win / self.playoutCount

def main():
    blackPlayer = NegaScoutBoardScoreAIPlayer(Turn.WHITE, 5)
    whitePlayer = SimpleMonteCarloPlayer(Turn.BLACK, 20)

    game = Game()
    board = Board()
    board.setup()
    winner = game.play(blackPlayer, whitePlayer, board, Turn.BLACK, True)

    print('BLACK = ' + type(blackPlayer).__name__)
    print('WHITE = ' + type(whitePlayer).__name__)
    print("Winner: " + str(winner))

main()
# EvalWinRate.main()
