import STcpClient
import numpy as np
import random

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
    most_hollow = 0
    for x in range(15):
        for y in range(15):
            if mapStat[x][y] == 0:
                adjacent_values = [
                    mapStat[i][j]
                    for i, j in [
                        (x - 1, y),
                        (x + 1, y),
                        (x, y - 1),
                        (x, y + 1),
                    ]  # 上下左右四个点
                    if 0 <= i < 15 and 0 <= j < 15  # 确保在地图范围内
                ]
                if -1 in adjacent_values or x == 0 or y == 0 or x == 14 or y == 14:
                    hollow = 0
                    for a in range(max(0, x - 1), min(15, x + 2)):
                        for b in range(max(0, y - 1), min(15, y + 2)):
                            if (a, b) != (x, y) and mapStat[a][b] == 0:
                                hollow += 1
                    if hollow > most_hollow:
                        most_hollow = hollow
                        init_pos = [x, y]
    return init_pos


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
    longest_move = step
    many_sheep = 0
    many_sheep_pos = []
    for x in range(15):
        for y in range(15):
            if mapStat[x][y] == playerID and sheepStat[x][y] >= 2:
                many_sheep_pos.append((x, y))
    many_sheep_pos.sort(key=lambda pos: sheepStat[pos[0]][pos[1]], reverse=True)
    
    for pos in many_sheep_pos:
        longest_dist = 0
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
            x, y = pos
            temp_x, temp_y = pos
            dx, dy = direction_moves[direction]
            while (
                0 <= temp_x + dx < 15
                and 0 <= temp_y + dy < 15
                and mapStat[temp_x + dx][temp_y + dy] == 0
            ):
                temp_x += dx
                temp_y += dy
                distance += 1

            if distance > longest_dist:
                longest_dist = distance
                m = int(sheepStat[x][y] / 8 + 0.75)
                longest_move = [(x, y), m, direction]
        if longest_dist == 0:
            continue
        else:
            break
    return longest_move


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

    STcpClient.SendStep(id_package, Step)
