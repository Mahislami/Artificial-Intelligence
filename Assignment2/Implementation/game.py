import random
import copy
import os
import math
import sys
import time

class GameError(AttributeError):
    pass

class Game:

    def __init__(self, n):
        self.size = n
        self.half_the_size = int(n/2)
        self.reset()

    def reset(self):
        self.board = []
        value = 'B'
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append(value)
                value = self.opponent(value)
            self.board.append(row)
            if self.size%2 == 0:
                value = self.opponent(value)

    def __str__(self):
        result = "  "
        for i in range(self.size):
            result += str(i) + " "
        result += "\n"
        for i in range(self.size):
            result += str(i) + " "
            for j in range(self.size):
                result += str(self.board[i][j]) + " "
            result += "\n"
        return result

    def valid(self, row, col):
        return row >= 0 and col >= 0 and row < self.size and col < self.size

    def contains(self, board, row, col, symbol):
        return self.valid(row,col) and board[row][col]==symbol

    def countSymbol(self, board, symbol):
        count = 0
        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == symbol:
                    count += 1
        return count

    def opponent(self, player):
        if player == 'B':
            return 'W'
        else:
            return 'B'

    def distance(self, r1, c1, r2, c2):
        return abs(r1-r2 + c1-c2)

    def makeMove(self, player, move):
        self.board = self.nextBoard(self.board, player, move)

    def nextBoard(self, board, player, move):
        r1 = move[0]
        c1 = move[1]
        r2 = move[2]
        c2 = move[3]
        next = copy.deepcopy(board)
        if not (self.valid(r1, c1) and self.valid(r2, c2)):
            raise GameError
        if next[r1][c1] != player:
            raise GameError
        dist = self.distance(r1, c1, r2, c2)
        if dist == 0:
            if self.openingMove(board):
                next[r1][c1] = "."
                return next
            raise GameError
        if next[r2][c2] != ".":
            raise GameError
        jumps = int(dist/2)
        dr = int((r2 - r1)/dist)
        dc = int((c2 - c1)/dist)
        for i in range(jumps):
            if next[r1+dr][c1+dc] != self.opponent(player):
                raise GameError
            next[r1][c1] = "."
            next[r1+dr][c1+dc] = "."
            r1 += 2*dr
            c1 += 2*dc
            next[r1][c1] = player
        return next

    def openingMove(self, board):
        return self.countSymbol(board, ".") <= 1

    def generateFirstMoves(self, board):
        moves = []
        moves.append([0]*4)
        moves.append([self.size-1]*4)
        moves.append([self.half_the_size]*4)
        moves.append([(self.half_the_size)-1]*4)
        return moves

    def generateSecondMoves(self, board):
        moves = []
        if board[0][0] == ".":
            moves.append([0,1]*2)
            moves.append([1,0]*2)
            return moves
        elif board[self.size-1][self.size-1] == ".":
            moves.append([self.size-1,self.size-2]*2)
            moves.append([self.size-2,self.size-1]*2)
            return moves
        elif board[self.half_the_size-1][self.half_the_size-1] == ".":
            pos = self.half_the_size -1
        else:
            pos = self.half_the_size
        moves.append([pos,pos-1]*2)
        moves.append([pos+1,pos]*2)
        moves.append([pos,pos+1]*2)
        moves.append([pos-1,pos]*2)
        return moves

    def check(self, board, r, c, rd, cd, factor, opponent):
        if self.contains(board,r+factor*rd,c+factor*cd,opponent) and \
           self.contains(board,r+(factor+1)*rd,c+(factor+1)*cd,'.'):
            return [[r,c,r+(factor+1)*rd,c+(factor+1)*cd]] + \
                   self.check(board,r,c,rd,cd,factor+2,opponent)
        else:
            return []

    def generateMoves(self, board, player):
        if self.openingMove(board):
            if player=='B':
                return self.generateFirstMoves(board)
            else:
                return self.generateSecondMoves(board)
        else:
            moves = []
            rd = [-1,0,1,0]
            cd = [0,1,0,-1]
            for r in range(self.size):
                for c in range(self.size):
                    if board[r][c] == player:
                        for i in range(len(rd)):
                            moves += self.check(board,r,c,rd[i],cd[i],1,
                                                self.opponent(player))
            return moves

    def playOneGame(self, p1, p2, show):
        self.reset()
        while True:
            if show:
                print(self)
                print("player B's turn")
            move = p1.getMove(self.board)
            if move == []:
                print("Game over")
                return 'W'
            try:
                self.makeMove('B', move)
            except GameError:
                print("Game over: Invalid move by", p1.name)
                print(move)
                print(self)
                return 'W'
            if show:
                print(move)
                print(self)
                print("player W's turn")
            move = p2.getMove(self.board)
            if move == []:
                print("Game over")
                return 'B'
            try:
                self.makeMove('W', move)
            except GameError:
                print("Game over: Invalid move by", p2.name)
                print(move)
                print(self)
                return 'B'
            if show:
                print(move)

    def playNGames(self, n, p1, p2, show):
        first = p1
        second = p2
        for i in range(n):
            print("Game", i)
            winner = self.playOneGame(first, second, show)
            if winner == 'B':
                first.won()
                second.lost()
                print(first.name, "wins")
            else:
                first.lost()
                second.won()
                print(second.name, "wins")
            first, second = second, first


class Player:
    name = "Player"
    wins = 0
    losses = 0
    def results(self):
        result = self.name
        result += " Wins:" + str(self.wins)
        result += " Losses:" + str(self.losses)
        return result
    def lost(self):
        self.losses += 1
    def won(self):
        self.wins += 1
    def reset(self):
        self.wins = 0
        self.losses = 0

    def initialize(self, side):
        abstract()

    def getMove(self, board):
        abstract()


class MinimaxPlayer(Game, Player):
    INDEX_INDEX = 0
    LEVEL_INDEX = 1
    #FATHER_NODE_INDEX = 2
    SCORE_INDEX = 2
    #BOARD_INDEX = 4
    CHILDREN_INDEX = 3
    MAXIMIZER = 'B'
    MINIMIZER = 'W'
    
    BOARD_SIZE = 8
    
    Max_level = 0
    
    level = 0
    
    #tempFilesPath = []
    
    read_node_offset = 0
    last_file_length = 0
    FILE_MAX_LENGTH = 100
    boardsFilePath = "boards" + str(BOARD_SIZE) + "_"
    
    nodes_saved = 0
    number_of_files = 0
    
    countAB = 0
    
    gameTree = []
    def __init__(self,n,level):
        self.Max_level = level
        print(self.Max_level)
        super().__init__(n)
    
    def initialize(self, side):
        self.side = side
        self.name = "Minimax"
        self.generateGameTree()
        start_time = time.time()
        #self.minimaxer(0,0,self.side)
        self.alphaBetaPruning(0,0,self.side,-math.inf,+math.inf)
        print("--- %s seconds ---" % (time.time() - start_time))
        #for node in self.gameTree:
        #    print(node)
        #print(self.side)
        
    def getMove(self, board):
        
        moves = self.generateMoves(board, self.side)
        index = self.boardIndex(self.boardsFilePath, self.generateBoard1d(board))
        maxScore = 0
        bestChildIndex = 0
        self.level = self.gameTree[index][self.LEVEL_INDEX]
        print(self.level)
        if self.level == self.Max_level:
            sys.exit()
        for i in range(len(self.gameTree[index][self.CHILDREN_INDEX])):
            if self.gameTree[self.gameTree[index][self.CHILDREN_INDEX][i]][self.SCORE_INDEX] > maxScore:
                maxScore = self.gameTree[self.gameTree[index][self.CHILDREN_INDEX][i]][self.SCORE_INDEX]
                bestChildIndex = i
        return moves[bestChildIndex]
        
    def generateGameTree(self):
        
        i = 0
        while True:
            tempPath = self.boardsFilePath + str(i) + ".txt"
            if os.path.exists(tempPath):
                os.remove(tempPath)
            else:
                break
            i += 1
        
        # Base level
        nodeIndex = 0
        self.appendBoard(self.boardsFilePath, self.generateBoard1d(self.board), nodeIndex)
        #score = self.evaluationFunction(self.board)
        self.gameTree += [self.generateNode(nodeIndex, 0, None)]
        
        
        
        # First level
        for move in self.generateFirstMoves(self.board):
            nodeIndex += 1
            lastBoard = copy.deepcopy(self.board)
            lastBoard[move[0]][move[1]] = '.'
            lastBoard1d = self.generateBoard1d(lastBoard)
            self.appendBoard(self.boardsFilePath, lastBoard1d, nodeIndex)
            self.gameTree[0][self.CHILDREN_INDEX].append(nodeIndex)
            score = self.evaluationFunction(lastBoard)
            if self.gameTree[0][self.LEVEL_INDEX] == self.Max_level - 1:
                self.gameTree += [self.generateNode(nodeIndex, self.gameTree[0][self.LEVEL_INDEX] + 1, score)]
            else:
                self.gameTree += [self.generateNode(nodeIndex, self.gameTree[0][self.LEVEL_INDEX] + 1, None)]
        
        if self.gameTree[0][self.CHILDREN_INDEX] == []:
            self.gameTree[0][self.SCORE_INDEX] = self.evaluationFunction(fatherBoard)
            
        
        # Second level
        for i in range(1, 5):
            fatherBoard = self.generateBoard2d(self.readBoard(self.boardsFilePath, i))
            for move in self.generateSecondMoves(fatherBoard):
                nodeIndex += 1
                lastBoard = copy.deepcopy(fatherBoard)
                lastBoard[move[0]][move[1]] = '.'
                lastBoard1d = self.generateBoard1d(lastBoard)
                self.appendBoard(self.boardsFilePath, lastBoard1d, nodeIndex)
                self.gameTree[i][self.CHILDREN_INDEX].append(nodeIndex)
                score = self.evaluationFunction(lastBoard)
                if self.gameTree[i][self.LEVEL_INDEX] == self.Max_level - 1:
                    self.gameTree += [self.generateNode(nodeIndex, self.gameTree[i][self.LEVEL_INDEX] + 1, score)]
                else:
                    self.gameTree += [self.generateNode(nodeIndex, self.gameTree[i][self.LEVEL_INDEX] + 1, None)]
            if self.gameTree[i][self.CHILDREN_INDEX] == []:
                self.gameTree[i][self.SCORE_INDEX] = self.evaluationFunction(fatherBoard)
        
        
        # Other levels
        i = 5
        self.level = 3
        while True:
            board2d = self.readBoard(self.boardsFilePath, i)
            if board2d == ['']:
                break
            fatherBoard = self.generateBoard2d(board2d)
            if self.gameTree[i][self.LEVEL_INDEX] == self.Max_level:
                break
            if self.gameTree[i][self.LEVEL_INDEX] % 2 == 0:
                treeSide = 'B'
            else:
                treeSide = 'W'
            for move in self.generateMoves(fatherBoard, treeSide):
                nodeIndex += 1
                lastBoard = copy.deepcopy(fatherBoard)
                lastBoard = self.nextBoard(lastBoard, treeSide, move)
                lastBoard1d = self.generateBoard1d(lastBoard)
                self.appendBoard(self.boardsFilePath, lastBoard1d, nodeIndex)
                self.gameTree[i][self.CHILDREN_INDEX].append(nodeIndex)
                score = self.evaluationFunction(lastBoard)
                if self.gameTree[i][self.LEVEL_INDEX] == self.Max_level - 1:
                    self.gameTree += [self.generateNode(nodeIndex, self.gameTree[i][self.LEVEL_INDEX] + 1, score)]
                else:
                    self.gameTree += [self.generateNode(nodeIndex, self.gameTree[i][self.LEVEL_INDEX] + 1, None)]
            if self.gameTree[i][self.CHILDREN_INDEX] == []:
                self.gameTree[i][self.SCORE_INDEX] = self.evaluationFunction(fatherBoard)
            
            #print(str(i) + " " + str(self.gameTree[i]))
            i += 1
        
        #for node in self.gameTree:
         #   print(node)
        self.number_of_files = len(self.gameTree) // self.FILE_MAX_LENGTH + 1
        return
        
    def minimaxer(self, level, index, side):
        
        if level == self.Max_level or self.gameTree[index][self.CHILDREN_INDEX] == []:
            return self.gameTree[index][self.SCORE_INDEX]
           
        if side == self.MAXIMIZER:
            maximum = -math.inf
            for child in self.gameTree[index][self.CHILDREN_INDEX]:
                a = self.minimaxer(self.gameTree[child][self.LEVEL_INDEX] , child, self.MINIMIZER)
                self.countAB += 1
                print(self.countAB)
                if a > maximum:
                    maximum = a
            self.gameTree[index][self.SCORE_INDEX] = maximum        
            return maximum
                
        if side == self.MINIMIZER:
            minimum = +math.inf
            for child in self.gameTree[index][self.CHILDREN_INDEX]:
                b = self.minimaxer(self.gameTree[child][self.LEVEL_INDEX], child, self.MAXIMIZER)
                if b < minimum:
                    minimum = b
            self.gameTree[index][self.SCORE_INDEX] = minimum       
            return minimum
            
    def alphaBetaPruning(self, level, index, side, alpha, beta):
        if level == self.Max_level or self.gameTree[index][self.CHILDREN_INDEX] == []:
            return self.gameTree[index][self.SCORE_INDEX]
            
        if side == self.MAXIMIZER:
            maximum = -math.inf
            for child in self.gameTree[index][self.CHILDREN_INDEX]:
                a = self.alphaBetaPruning(self.gameTree[child][self.LEVEL_INDEX] , child, self.MINIMIZER, alpha, beta)
                self.countAB += 1
                print(self.countAB)
                if a > maximum:
                    maximum = a
                alpha = max(alpha, maximum)
                if beta <= alpha:  
                    break 
            self.gameTree[index][self.SCORE_INDEX] = maximum        
            return maximum
                
        if side == self.MINIMIZER:
            minimum = +math.inf
            for child in self.gameTree[index][self.CHILDREN_INDEX]:
                b = self.alphaBetaPruning(self.gameTree[child][self.LEVEL_INDEX], child, self.MAXIMIZER, alpha, beta)
                if b < minimum:
                    minimum = b
                beta = min(beta, minimum)
                if beta <= alpha:  
                    break 
            self.gameTree[index][self.SCORE_INDEX] = minimum       
            return minimum
        
    def evaluationFunction(self, board):
        return len(self.generateMoves(board, self.MAXIMIZER)) - len(self.generateMoves(board, self.MINIMIZER))
        
    def getLevel(self, board):
        level = 0
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if board[i][j] == '.':
                    level += 1
        return level
        
    def generateBoard1d(self, board):
        board1d = []
        for i in range(self.BOARD_SIZE):
            board1d += board[i]
        return board1d
        
    def generateBoard2d(self, board1d):
        board2d = []
        for i in range(self.BOARD_SIZE):
            board2d += [board1d[(i*self.BOARD_SIZE):((i+1)*self.BOARD_SIZE)]]
        return board2d
        
    def appendBoard(self, path, board1d, index):
        a = index // self.FILE_MAX_LENGTH
        path = path + str(a) + ".txt"
        file = open(path, "a")
        file.write(str(board1d) + "\n")
        
    def appendNode(self, node):
        path = self.tempFilesPath[len(self.tempFilesPath) - 1]
        file = open(path, "a")
        file.write(str(node[self.INDEX_INDEX]) + "\n")
        file.write(str(node[self.LEVEL_INDEX]) + "\n")
        file.write(str(node[self.FATHER_NODE_INDEX]) + "\n")
        file.write(str(node[self.SCORE_INDEX]) + "\n")
        for i in range(self.BOARD_SIZE):
            file.write(str(node[self.BOARD_INDEX][i]) + "\n")
        file.write(str(node[self.CHILDREN_INDEX]) + "\n")
        file.write("\n")
        self.last_file_length += 1
        if self.last_file_length == self.FILE_MAX_LENGTH:
            self.number_of_files += 1
            self.tempFilesPath += ["temp" + str(self.BOARD_SIZE) + "_" + str(self.number_of_files) + ".txt"]
            self.last_file_length = 0
        file.close()
        
    def saveNode(self, path, node):
        file = open(path, "a")
        file.write(str(node[self.INDEX_INDEX]) + "\n")
        file.write(str(node[self.LEVEL_INDEX]) + "\n")
        file.write(str(node[self.FATHER_NODE_INDEX]) + "\n")
        file.write(str(node[self.SCORE_INDEX]) + "\n")
        for i in range(self.BOARD_SIZE):
            file.write(str(node[self.BOARD_INDEX][i]) + "\n")
        file.write(str(node[self.CHILDREN_INDEX]) + "\n")
        file.write("\n")
        self.nodes_saved += 1
        if self.nodes_saved % self.FILE_MAX_LENGTH == 0:
            self.tempFilesPath.pop(0)
            self.read_node_offset += self.FILE_MAX_LENGTH
        file.close()
        
    def readBoard(self, path, index):
        fileIndex = index // self.FILE_MAX_LENGTH
        path = path + str(fileIndex) + ".txt"
        file = open(path, "r")
        lineNumber = index % self.FILE_MAX_LENGTH
        i = 0
        line = file.readline()
        output = None
        while True:
            if i == lineNumber:
                return line.rstrip('\n').strip("']['").split("', '")
            i += 1
            line = file.readline()
        return output
        
    def boardIndex(self, path, board1d):
        index = 0
        found = False
        for i in range(self.number_of_files):
            path = path + str(i) + ".txt"
            if os.path.exists(path):
                file = open(path, "r")
                j = 0
                line = file.readline()
                while True:
                    if board1d == line.rstrip('\n').strip("']['").split("', '"):
                        found = True
                        index = i * self.FILE_MAX_LENGTH + j
                        break
                    j += 1
                    line = file.readline()
                if found == True:
                    break
                file.close()
            else:
                break
        if found == False:
            return None
        else:
            return index
        
    def readNode(self, index):
        path = self.tempFilesPath[0]
        file = open(path, "r")
        nodeIndex = (self.CHILDREN_INDEX + self.BOARD_SIZE + 1) * (index - self.read_node_offset)
        node = []
        node += ["Index"]
        node += ["Level"]
        node += ["Father Index"]
        node += ["Score"]
        node += ["Board"]
        node += ["Children"]
        i = 0
        line = file.readline()
        while line:
            if i == nodeIndex:
                node[self.INDEX_INDEX] = index
                level = file.readline().rstrip('\n')
                node[self.LEVEL_INDEX] = level
                fatherIndex = file.readline().rstrip('\n')
                node[self.FATHER_NODE_INDEX] = fatherIndex
                score = file.readline().rstrip('\n')
                node[self.SCORE_INDEX] = score
                board = []
                for j in range(self.BOARD_SIZE):
                    board += [file.readline().rstrip('\n').strip("']['").split("', '")]
                node[self.BOARD_INDEX] = board
                children = file.readline().rstrip('\n')
                if children != "Children":
                    children = children.strip("']['").split("', '")
                node[self.CHILDREN_INDEX] = children
                break
            for j in range((self.CHILDREN_INDEX + self.BOARD_SIZE + 1)):
                i += 1
                line = file.readline()
        file.close()
        return node
        
    def deleteNode(self, path, index):
        
        with open(path, 'r') as file:
            with open("temp.txt", 'w') as temp:
                line = file.readline()
                i = 0
                while line:
                    if i == (self.CHILDREN_INDEX + self.BOARD_SIZE + 1) * (index - self.read_node_offset):
                        for j in range((self.CHILDREN_INDEX + self.BOARD_SIZE + 1)):
                            line = file.readline()
                            i += 1
                        continue
                    temp.write(line)
                    line = file.readline()
                    i += 1
            
        os.remove(path)
        os.rename('temp.txt', path)
        self.read_node_offset += 1
        
    def generateBaseNode(self):
        firstNode = []
        firstNode += ["Index"]
        firstNode += ["Score"]
        firstNode += ["Children"]
        
        firstNode[self.INDEX_INDEX] = 0
        firstNode[self.SCORE_INDEX] = None
        firstNode[self.CHILDREN_INDEX] = []
        
        return firstNode
        
    def generateFirstAndSecondLevelNode(self, fatherNode, fatherIndex, index, level, move):
        node = []
        
        node += ["Index"]
        node += ["Level"]
        node += ["Father Index"]
        node += ["Score"]
        node += ["Board"]
        node += ["Children"]
        
        node[self.INDEX_INDEX] = index
        node[self.LEVEL_INDEX] = level
        node[self.FATHER_NODE_INDEX] = fatherIndex
        
        next = copy.deepcopy(fatherNode[self.BOARD_INDEX])
        next[move[0]][move[1]] = "."
        node[self.BOARD_INDEX] = next
        
        return node
        
    def generateNode(self, index, level, score):
        node = []
        node += ["Index"]
        node += ["Level"]
        node += ["Score"]
        node += ["Children"]
        
        node[self.INDEX_INDEX] = index
        node[self.LEVEL_INDEX] = level
        node[self.SCORE_INDEX] = score
        node[self.CHILDREN_INDEX] = []
        
        return node
        
    def insertChild(self, index, child):
        print(' ')
    
    def insertScore(self, node, score):
        node[self.SCORE_INDEX] = score
        return node
        
        
        
class SimplePlayer(Game, Player):
    def initialize(self, side):
        self.side = side
        self.name = "Simple"
    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        n = len(moves)
        if n == 0:
            return []
        else:
            return moves[0]

class RandomPlayer(Game, Player):
    def initialize(self, side):
        self.side = side
        self.name = "Random"
    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        n = len(moves)
        if n == 0:
            return []
        else:
            return moves[random.randrange(0, n)]

class HumanPlayer(Game, Player):
    def initialize(self, side):
        self.side = side
        self.name = "Human"
    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        while True:
            print("Possible moves:", moves)
            n = len(moves)
            if n == 0:
                print("You must concede")
                return []
            index = input("Enter index of chosen move (0-"+ str(n-1) +
                          ") or -1 to concede: ")
            try:
                index = int(index)
                if index == -1:
                    return []
                if 0 <= index <= (n-1):
                    print("returning", moves[index])
                    return moves[index]
                else:
                    print("Invalid choice, try again.")
            except Exception as e:
                print("Invalid choice, try again.")
            

n = 8
if __name__ == '__main__':
    game = Game(n)
    human1 = MinimaxPlayer(n,4)
    human1.initialize('B')
    human2 = HumanPlayer(n)
    human2.initialize('W')
    #human2.generateGameTree()
    
    # computer1 = RandomPlayer(8)
    # computer1.initialize('W')
    game.playOneGame(human1, human2, True)
