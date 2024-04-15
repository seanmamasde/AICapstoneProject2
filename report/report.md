---
title: "Artificial Intelligence Capstone: Homework 2"
author: 
    - Team No. #19
	- 110550032 文玠敦
	- 110550038 莊書杰
	- 110550074 馬楷翔
date: \today{}
CJKmainfont: "Microsoft YaHei"
--- 

## Introduction

In this assignment we designed game agents for a board game called Battle Sheep. This project is divided into 4 parts, each part is a slightly different variant of the original board game (most of the rules remain unchanged), and presents a different set of criteria as challenges for our agent, which is as follows:

1.  The Basic Form
    The game is conducted on a `12*12` board, which then will randomly generate a connected portion consists of `64` playable cells. Each of the 4 players gets `16` sheep initially.
2.  Overcrowded Bigger Board
    Each player now has `32` sheep to begin the game with. Also the field is now `100` cells in total chosen within a `15*15` board.
3.  Imperfect Information
    A player (which is out agent) does not know the number of sheep in cells occupied by other players.
4.  Cooperative Play
    The 4 players form 2 teams: with player 1 & 3 on the same team and 2 & 4 on the other.

