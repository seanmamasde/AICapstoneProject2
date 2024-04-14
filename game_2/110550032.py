import STcpClient
import numpy as np
import random
import copy
from time import time

BOARD_SIZE = 15
SHEEP_NUM = 32
time_limit = 3 * 0.95
#MAX_ITER = 100
final_score = [1, 0.5, 0.3, 0.1]
all_dir = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

class Node:
    def __init__(self, id, split, sheep, action, mapStat, sheepStat, init):
        self.id = id
        self.split = split
        self.sheep = sheep
        self.action = action
        self.mapstat = mapStat
        self.sheepstat = sheepStat
        self.visits = 0
        self.score = 0
        self.children = []
        self.parent = []
        self.init = init

""" functions for calculate score """   

def dfs(mapStat, row, col, visited, id):
    grid = mapStat
    stack = [(row, col)]
    area = 0
    
    while stack:
        r, c = stack.pop()
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and not visited[r][c] and grid[r][c] == id:
            visited[r][c] = True
            area += 1
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
    
    return area

def find_connected_components(mapStat, id):
    grid = mapStat
    rows = BOARD_SIZE
    cols = BOARD_SIZE
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    areas = []
    
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and grid[i][j] == id:
                area = dfs(grid, i, j, visited, id)
                areas.append(area)
    
    return areas

""" funcions for init pos """

def init_choices(mapStat):
    all_choices = []
    visited = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            visited[i][j] = True
            if (i == 0 or i == BOARD_SIZE - 1 or j == 0 or j == BOARD_SIZE - 1) and mapStat[i][j] == 0:
                #print("get!!")
                all_choices.append((i, j))
            if mapStat[i][j] == -1:
                if i - 1 >= 0:
                    if mapStat[i - 1][j] == 0 and visited[i - 1][j] == False:
                        visited[i - 1][j] = True
                        all_choices.append((i - 1, j))
                if i + 1 < BOARD_SIZE:
                    if mapStat[i + 1][j] == 0 and visited[i + 1][j] == False:
                        visited[i + 1][j] = True
                        all_choices.append((i + 1, j))
                if j - 1 >= 0:
                    if mapStat[i][j - 1] == 0 and visited[i][j - 1] == False:
                        visited[i][j - 1] = True
                        all_choices.append((i, j - 1))
                if j + 1 < BOARD_SIZE:
                    if mapStat[i][j + 1] == 0 and visited[i][j + 1] == False:
                        visited[i][j + 1] = True
                        all_choices.append((i, j + 1))
    #print("1: ", all_choices)
    return all_choices

""" functions for get step """

def valid_dir(mapStat, sheep):
    dirs = []
    for i in range(9):
        if sheep[0] + all_dir[i][0] >= 0 and sheep[0] + all_dir[i][0] < BOARD_SIZE and sheep[1] + all_dir[i][1] >= 0 and sheep[1] + all_dir[i][1] < BOARD_SIZE:
            if mapStat[sheep[0] + all_dir[i][0]][sheep[1] + all_dir[i][1]] == 0:
                dirs.append(i + 1)
    return dirs

def movable_sheeps(mapStat, sheepStat, id):
    movable = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if mapStat[i][j] == id:
                if len(valid_dir(mapStat, (i, j))) != 0 and sheepStat[i][j] > 1:
                    movable.append((i, j))
    return movable

def action(mapStat, sheepStat, id):
    sheep = random.choice(movable_sheeps(mapStat, sheepStat, id))
    split = random.randint(1, int(sheepStat[sheep[0]][sheep[1]]) - 1)
    act = random.choice(valid_dir(mapStat, sheep))
    return split, sheep, act

def updata_map(mapStat, sheepStat, split, sheep, act):
    new_mapstat = copy.deepcopy(mapStat)
    new_sheepstat = copy.deepcopy(sheepStat)
    temp = sheep
    new_loc = (sheep[0] + all_dir[act - 1][0], sheep[1] + all_dir[act - 1][1])
    while new_loc[0] >= 0 and new_loc[0] < BOARD_SIZE and new_loc[1] >= 0 and new_loc[1] < BOARD_SIZE:
        #print("here5")
        if mapStat[new_loc[0]][new_loc[1]] == 0:
            temp = new_loc
            new_loc = (temp[0] + all_dir[act - 1][0], temp[1] + all_dir[act - 1][1])
        else:
            break
    new_sheepstat[sheep[0]][sheep[1]] -= split
    new_sheepstat[temp[0]][temp[1]] += split
    new_mapstat[temp[0]][temp[1]] = mapStat[sheep[0]][sheep[1]]

    return new_mapstat, new_sheepstat

def get_rank(mapStat):
    scores = []
    for i in range(4):
        areas = find_connected_components(mapStat, i + 1)
        sum = 0
        for area_size in areas:
            sum += area_size**1.25
        score = np.round(sum)
        scores.append(score)
    
    sorted_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    rank = [0] * len(scores)
    for i, idx in enumerate(sorted_indices):
        rank[idx] = i
    
    return rank

def UCTValue(total_visits, node_visits, node_score):
    if node_visits == 0:
        return float('inf')
    return (node_score / node_visits) + np.sqrt(2 * np.log(total_visits) / node_visits)

""" functions for MCTS algorithm """

def Select(root):
    selected_node = None
    best_value = -float('inf')
    for child in root.children:
        value = UCTValue(root.visits, child.visits, child.score)
        if value > best_value:
            best_value = value
            selected_node = child
    return selected_node

def Expand(node):
    count = 0
    id = node.id
    mapStat = node.mapstat
    sheepStat = node.sheepstat
    if node.init:
        node.init = False
        all_choices = init_choices(mapStat)
        num = 1
        for init_choice in all_choices:
            #print("\n\n=====================\n",mapStat)
            new_mapStat = copy.deepcopy(mapStat)
            new_sheepStat = copy.deepcopy(sheepStat)
            new_mapStat[init_choice[0]][init_choice[1]] = id
            new_sheepStat[init_choice[0]][init_choice[1]] = SHEEP_NUM
            #if num <= 2:
                #print("\n",new_mapStat)
                #print("\n",new_sheepStat)
                
            #print("\n",mapStat)
            #print("\n",new_mapStat)
            new_id = id % 4 + 1
            bool_value = False
            if new_id != 1:
                bool_value = True
            new_node = Node(new_id, -1, init_choice, [], new_mapStat, new_sheepStat, bool_value)
            node.children.append(new_node)
            new_node.parent.append(node)
            count += 1
            num += 1
    else:
        for sheep in movable_sheeps(mapStat, sheepStat, id):
            for split in range(1, int(sheepStat[sheep[0]][sheep[1]])):
                for act in valid_dir(mapStat, sheep):
                    new_mapStat, new_sheepStat = updata_map(mapStat, sheepStat, split, sheep, act)
                    new_id = id % 4 + 1
                    new_node = Node(new_id, split, sheep, act, new_mapStat, new_sheepStat, False)
                    node.children.append(new_node)
                    new_node.parent.append(node)
                    count += 1
    return count

def Simulate(node):
    mapStat = node.mapstat
    sheepStat = node.sheepstat        
    endgame = False
    flag = node.init
    while not endgame:
        #print("here0")
        endgame = True
        if flag:
            #print("here1")
            for i in range(4):
                endgame = False
                id = (node.id + i - 1) % 4 + 1
                all_choices = init_choices(mapStat)
                #print("2: ", all_choices)
                init_choice = random.choice(all_choices)
                mapStat[init_choice[0]][init_choice[1]] = id
                sheepStat[init_choice[0]][init_choice[1]] = SHEEP_NUM
                if id == 4:
                    flag = False
                    break
        else:
            #print("here2")
            for i in range(4):
                id = (node.id + i - 1) % 4 + 1 
                if len(movable_sheeps(mapStat, sheepStat, id)) > 0:
                    endgame = False
                    split, sheep, act = action(mapStat, sheepStat, id)
                    new_mapStat, new_sheepStat = updata_map(mapStat, sheepStat, split, sheep, act)
                    mapStat = copy.deepcopy(new_mapStat)
                    sheepStat = copy.deepcopy(new_sheepStat)
    rank = get_rank(mapStat)
    score = [0, 0, 0, 0]
    for i in range(4):
        score[i] = final_score[rank[i]]

    return score

def Backpropagate(node, score):
    while len(node.parent) > 0:
        #print("node.id", node.id)
        node.visits += 1
        node.score += score[(node.id - 2) % 4]
        node = node.parent[0]
    node.visits += 1
    node.score += score[node.id - 1]

def MTCS(playerID, mapStat, sheepStat):
    
    flag = True
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if mapStat[i][j] == playerID:
                flag = False
    root = Node(playerID, -1, (-1, -1), [], mapStat, sheepStat, flag)
    #max_iterations = MAX_ITER

    time_count = 0
    start_time = time()
    while time() - start_time < time_limit:
        #print(mapStat)
        #print(sheepStat)
        #print("iter: ", _, "\n")
        node = root
        # Selection phase
        while node.children:
            node = Select(node)
        #print("4")
        # Expansion phase
        count = Expand(node)
        #print("3")
        if count > 0:
            expanded_node = random.choice(node.children)
            score = Simulate(expanded_node)
            Backpropagate(expanded_node, score)
        else:
            score = Simulate(node)
            Backpropagate(node, score)
        time_count += 1
        
    # Choose the best action based on the most visited child of the root
    best_child = max(root.children, key=lambda x: x.visits)
    print(time_count)
    if flag:
        return best_child.sheep
    else:
        return best_child.split, best_child.sheep, best_child.action

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''


def InitPos(mapStat):
    init_pos = [0, 0]
    sheepStat = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    init_choice = MTCS(1, mapStat, sheepStat)
    init_pos = [init_choice[0], init_choice[1]]
    return init_pos
    


'''
    產出指令
    
    input: 
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 15*15矩陣, 
              0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~32, 為 15*15矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
            4 X 6
            7 8 9
'''
def GetStep(playerID, mapStat, sheepStat):
    step = [(0, 0), 0, 1]
    
    split, sheep, action = MTCS(playerID, mapStat, sheepStat)
    step = [sheep, split, action]
    return step


# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
