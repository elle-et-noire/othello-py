from enum import Enum
import copy

class Stone(Enum):
    EMPTY = 0
    BLACK = -1
    WHITE = 1
    WALL = 334

    def flip(self):
        return Stone(self.value * -1)

class Position():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equals(self, p):
        if self.x == p.x and self.y == p.y:
            return True
        return False

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


class Turn(Enum):
    FIRST = -1
    SECOND = 1

    def flip(self):
        return Turn(self.value * -1)

    def stone(self):
        #先手は黒、後手は白
        return Stone(self.value)

class HumanPlayer():
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


class SimpleAIPlayer():
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

    def __init__(self, turn):
        self.turn = turn

    def play(self, board):
        pos = Position(0, 0)
        maxVal = -float('inf')

        for p in board.findPuttableHands(self.turn.stone()):
            if maxVal < self.EVAL_VALUES[p.y - 1][p.x - 1]:
                maxVal = self.EVAL_VALUES[p.y - 1][p.x - 1]
                pos = p

        return p

class EvalResult():
    def __init__(self, score, position):
        self.score = score
        self.position = position

class MinMaxAIPlayer(SimpleAIPlayer):
    def __init__(self, turn, maxDepth):
        super().__init__(turn)
        self.maxDepth = maxDepth

    def play(self, board):
        return self.eval(board, self.maxDepth, self.turn, 0).position

    def eval(self, board, restDepth, currentTurn, scoreSum):
        if restDepth == 0:
            return EvalResult(scoreSum if currentTurn == self.turn else -scoreSum, None)

        puttables = board.findPuttableHands(currentTurn.stone())

        if len(puttables) == 0:
            score = -self.eval(board, restDepth - 1, currentTurn.flip(), scoreSum).score
            return EvalResult(score, None)

        maxScore = -float('inf')
        selected = None

        for p in puttables:
            b = board.clone()
            b.put(p.x, p.y, currentTurn.stone())
            scoreDiff = (1 if currentTurn == self.turn else -1) * self.EVAL_VALUES[p.y - 1][p.x - 1]
            score = -self.eval(b, restDepth - 1, currentTurn.flip(), scoreSum + scoreDiff).score
            if maxScore < score:
                maxScore = score
                selected = p
        return EvalResult(maxScore, selected)


class AlphaBetaMinMaxAIPlayer(SimpleAIPlayer):
    def __init__(self, turn, maxDepth):
        super().__init__(turn)
        self.maxDepth = maxDepth

    def play(self, board):
        return self.eval(board, self.maxDepth, self.turn, 0, -float('inf'), float('inf')).position

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
    
class FixedStones():
    def __init__(self):
        self.fixedStone = self.calculate()

    def getNumFixedStones(self, board, stone):
        upper = []
        lower = []
        left = []
        right = []

        for i in range(1,9):
            upper.append(board.get(i, 1))
            lower.append(board.get(i, Board.HEIGHT))
            left.append(board.get(1, i))
            right.append(board.get(Board.WIDTH, i))

        if stone != Board.BLACK:
            for i in range(8):
                upper[i] = upper[i].flip()
                lower[i] = lower[i].flip()
                left[i] = left[i].flip()
                right[i] = right[i].flip()

        if board.get(1, 1) == stone:
            count -= 1
        if board.get(1, Board.HEIGHT) == stone:
            count -= 1
        if board.get(Board.WIDTH, 1) == stone:
            count -= 1
        if board.get(Board.WIDTH, Board.HEIGHT) == stone:
            count -= 1

        return count

    def calculate(self):
        return [self.calculateIter(self.fromInt(i), 1) for i in range(3^8+1)]

    def calculateIter(self, ith, visited):
        if visited[ith] >= 0:
            return visited[ith]

        board = self.fromInt(ith)
        foundEmpty = False
        result = 100

        for x in range(1, Board.WIDTH+1):
            if board.get(x, 1) != Stone.EMPTY:
                continue

            foundEmpty = True

            b = board.clone()
            b.put(x, 1, Stone.BLACK)
            result = min(result, self.calculateIter(self.toInt(b.getHorizontal(1)), visited))

            b = board.clone()
            b.put(x, 1, Stone.WHITE)
            result = min(result, self.calculateIter(self.toInt(b.getHorisontal(1)), visited))

        if foundEmpty:
            visited[ith] = result
            return result

        numBlackStone = 0
        for i in range(1, Board.WIDTH+1):
            if board.get(i, 1) == Stone.BLACK:
                numBlackStone += 1

        visited[ith] = numBlackStone
        return numBlackStone

    def fromInt(self, idx):
        board = Board()
        for x in range(8, 0, -1):
            board.set(x, 1, Stone.values()[idx % 3])
            idx /= 3

        return board

    def toInt(self, stones):
        result = 0
        for i in range(len(stones)):
            result = result * 3 + stones[i].ordinal()

        return result
    
class AdvancedEvaluator():
    def __init__(self, turn, currentOpenness):
        self.turn = turn
        self.currentOpenness = currentOpenness

    def willPut(self, board, x, y, stone):
        for i in range(len(Board.DX)):
            dx = Board.DX[i]
            dy = Board.DY[i]
            count = board.countFlippable(x, y, stone, dx, dy)
            for j in range(1, count+1):
                if self.turn.stone() == stone:
                    self.currentOpenness += self.countOpenness(board, x + dx * j, y + dy * j)
                else:
                    self.currentOpenness += self.countOpenness(board, x + dx * j, y + dy * j)

    def countOpenness(self, board, x, y):
        count = 0
        for i in range(len(Board.DX)):
            dx = Board.DX[i]
            dy = Board.DY[i]
            if board.get(x + dx, y + dy) == Stone.EMPTY:
                count += 1
        return count

    def score(self, board, currentTurn):
        v = -self.currentOpenness * 3
        v += FixedStones.getNumFixedStones(board, self.turn.stone()) * 20
        v -= FixedStones.getNumFixedStones(board, self.turn.flip().stone()) * 20
        v += len(board.findPuttableHands(self.turn.stone()))
        v -= len(board.findPuttableHands(self.turn.flip().stone()))

        return v if currentTurn == self.turn else -v

def main():
    board = Board()
    board.setup()
    turn = Turn.FIRST
    hasPassed = False
    firstPlayer = SimpleAIPlayer(Turn.FIRST)
    secondPlayer = AlphaBetaMinMaxAIPlayer(Turn.SECOND, 5)

    while True:
        board.show()

        if not board.isPuttableSomewhere(turn.stone()):
            if hasPassed:
                break
            hasPassed = True
            turn = turn.flip()
            continue

        hasPassed = False
        p = Position(0, 0)
        if turn == Turn.FIRST:
            p = firstPlayer.play(board)
        else:
            p = secondPlayer.play(board)

        board.put(p.x, p.y, turn.stone())
        turn = turn.flip()

    print('BLACK = ' + str(board.countStone(Stone.BLACK)))
    print('WHITE = ' + str(board.countStone(Stone.WHITE)))


