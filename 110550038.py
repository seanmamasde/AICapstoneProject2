import STcpClient
import numpy as np
import random

class Node:
    def __init__(self, playerID, mapStat, sheepStat, action):
        self.playerID = playerID
        self.mapStat = mapStat
        self.sheepStat = sheepStat
        self.children = []
        self.action = action

    def generate_children(self):
        # 生成所有可能的子節點，代表玩家在當前狀態下的所有可能行動
        for x in range(12):
            for y in range(12):
                if self.mapStat[x][y] == self.playerID and self.sheepStat[x][y] >= 2:
                    # 對羊的分割數量進行迭代
                    max_dist = -1
                    best_direction = -1
                    for direction in range(1, 10):
                        # 對所有可能的移動方向進行迭代
                        direction_moves = {
                            1: (-1, -1),
                            2: (0, -1),
                            3: (1, -1),
                            4: (-1, 0),
                            5: (0, 0),
                            6: (1, 0),
                            7: (-1, 1),
                            8: (0, 1),
                            9: (1, 1),
                        }
                        distance = 0

                        # 獲取方向的位移
                        temp_x = x
                        temp_y = y
                        dx, dy = direction_moves[direction]

                        # 從當前位置開始移動，直到碰到障礙物或其他玩家的格子，並且沿路更新格子
                        while 0 <= temp_x + dx < 12 and 0 <= temp_y + dy < 12 and mapStat[temp_x + dx][temp_y + dy] == 0:
                            temp_x += dx
                            temp_y += dy
                            distance += 1

                        if distance > max_dist:
                            max_dist = distance
                            best_direction = direction
                    # 創建新的地圖狀態和羊群狀態
                    new_map_stat = [row[:] for row in self.mapStat]
                    new_sheep_stat = [row[:] for row in self.sheepStat]
                    # 在新的地圖狀態中更新移動後的位置
                    new_x, new_y = move_sheep(
                        x, y, best_direction, new_map_stat
                    )
                    
                    # 更新新位置的羊群數量
                    m = int(self.sheepStat[x][y] * 3 / 4 - 0.5)
                    new_sheep_stat[new_x][new_y] += m
                    new_map_stat[new_x][new_y] = self.playerID
                    new_sheep_stat[x][y] -= m
                    # 創建新節點並添加到子節點列表中
                    new_node = Node(
                        (self.playerID % 4) + 1,
                        new_map_stat,
                        new_sheep_stat,
                        [(x, y), m, best_direction],
                    )
                    self.children.append(new_node)


def move_sheep(x, y, direction, mapStat):
    # 定義所有可能的方向對應的位移
    direction_moves = {
        1: (-1, -1),
        2: (0, -1),
        3: (1, -1),
        4: (-1, 0),
        5: (0, 0),
        6: (1, 0),
        7: (-1, 1),
        8: (0, 1),
        9: (1, 1),
    }

    # 獲取方向的位移
    dx, dy = direction_moves[direction]
    temp_x = x
    temp_y = y

    # 從當前位置開始移動，直到碰到障礙物或其他玩家的格子，並且沿路更新格子
    while 0 <= temp_x + dx < 12 and 0 <= temp_y + dy < 12 and mapStat[temp_x + dx][temp_y + dy] == 0:
        temp_x += dx
        temp_y += dy

    # 返回新的 x 和 y 座標
    return temp_x, temp_y


def build_game_tree(node, depth):
    if depth == 0:
        return
    node.generate_children()
    for child in node.children:
        build_game_tree(child, depth - 1)


def evaluate_node(node):
    # 計算玩家的分數
    player_score = calculate_player_score(node.mapStat, node.playerID)
    # 如果達到第四層或節點沒有子節點，返回該節點的分數
    if len(node.children) == 0:
        return player_score
    # 否則，遞迴計算子節點的分數並返回最大值
    max_child_score = float("-inf")
    for child in node.children:
        child_score = evaluate_node(child)
        max_child_score = max(max_child_score, child_score)
    return max_child_score


# 在第四層找到最高分的行動方式
def find_best_action(node):
    best_action = None
    max_score = float("-inf")
    for child in node.children:
        child_score = evaluate_node(child)
        if child_score > max_score:
            max_score = child_score
            best_action = child.action
    return best_action


# 計算玩家的分數
def calculate_player_score(mapStat, playerID):
    # 計算玩家佔領的所有區域大小的總和，然後將其1.25次方
    player_regions = []
    visited = [[False] * 12 for _ in range(12)]
    for i in range(12):
        for j in range(12):
            if mapStat[i][j] == playerID and visited[i][j] == False:
                region_size = calculate_region_size(mapStat, i, j, playerID, visited)
                player_regions.append(region_size**1.25)
    # 將所有區域的1.25次方後的大小相加，並四捨五入取整數
    player_score = round(sum(player_regions))
    return player_score


# 計算連通區域的大小
def calculate_region_size(mapStat, x, y, playerID, visited):
    # 使用深度優先搜索計算連通區域的大小
    def dfs(i, j):
        if (
            i < 0
            or i >= 12
            or j < 0
            or j >= 12
            or visited[i][j]
            or mapStat[i][j] != playerID
        ):
            return 0
        visited[i][j] = True
        size = 1
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            size += dfs(i + dx, j + dy)
        return size

    return dfs(x, y)


# 在第四層找到最高分的行動方式
def find_best_action_at_depth_four(playerID, mapStat, sheepStat):
    root_node = Node(playerID, mapStat, sheepStat, None)
    build_game_tree(root_node, depth=4)
    return find_best_action(root_node)

"""
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
"""

def InitPos(mapStat):
    init_pos = [0, 0]
    """
        Write your code here

    """
    while True:
        x = random.randint(0, 11)
        y = random.randint(0, 11)
        if mapStat[x][y] == 0:
            # Check if at least one adjacent cell is -1 or if the position is at the border
            adjacent_values = [
                mapStat[i][j]
                for i in range(max(0, x - 1), min(12, x + 2))
                for j in range(max(0, y - 1), min(12, y + 2))
                if (i != x or j != y)
            ]
            if -1 in adjacent_values or x == 0 or y == 0 or x == 11 or y == 11:
                return [x, y]


"""
    產出指令
    
    input: 
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 
              0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
            4 X 6
            7 8 9
"""


def GetStep(playerID, mapStat, sheepStat):
    step = [(0, 0), 0, 1]
    """
    Write your code here
    
    """
    step = find_best_action_at_depth_four(playerID, mapStat, sheepStat)
    return step

def IsStepValid(Step, mapStat, playerID, sheepStat):
    (x, y), m, direction = Step

    if not (0 <= x < 12 and 0 <= y < 12):
        print(Step, 1)
        return False
    
    if mapStat[x][y] != playerID:
        print(Step, 2)
        return False
    
    direction_moves = {
        1: (-1, -1),
        2: (0, -1),
        3: (1, -1),
        4: (-1, 0),
        5: (0, 0),
        6: (1, 0),
        7: (-1, 1),
        8: (0, 1),
        9: (1, 1),
    }

    dx, dy = direction_moves[direction]

    new_x, new_y = x + dx, y + dy

    if not (0 <= new_x < 12 and 0 <= new_y < 12):
        print(Step, 3)
        return False
    
    if mapStat[new_x][new_y] == -1:
        print(Step, 4)
        return False
    
    if sheepStat[x][y] < 1:
        print(Step, 5)
        return False
    
    return True

# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while True:
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    print(mapStat)
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)
    # if IsStepValid(Step, mapStat, playerID, sheepStat):
    STcpClient.SendStep(id_package, Step)
    # else:
    #     print(mapStat)
    #     print(sheepStat)
    #     STcpClient.SendStep(id_package, [])
