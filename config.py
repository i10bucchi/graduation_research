#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml

# モデル
NUM_PLAYERS = 100
MAX_REP = 21
S = 1
MAX_TURN = 100
MAX_STEP = 10000
COMUNITY_MOVE_TERM = 20
LEADER_SAMPLING_TERM = 100

# カラム
NUM_COLUMN = 21

## 共有
COL_PLAYERID = 0
COL_GAME_REWARD = 1
COL_COMUNITY_REWARD = 2
COL_ROLE_REWARD = 3
COL_ROLE = 4
COL_QrLEADER = 5
COL_QrMEMBERS = 6
COL_ANUM = 7

## 所属コミュニティー
COL_COMUNITY = 8

## 成員用
COL_AC = 9
COL_AS = 10
COL_Qa00 = 11 # C / S
COL_Qa01 = 12
COL_Qa10 = 13
COL_Qa11 = 14

## 制裁者用
COL_APC = 15
COL_APS = 16
COL_Qap00 = 17 # PD / PnS
COL_Qap01 = 18
COL_Qap10 = 19
COL_Qap11 = 20

# パラメータ
COST_C_LIST = [4]
COST_S_LIST = [2]
COST_P_LIST = [2]
SP_LIST = [4]
PUNISH_SIZE_LIST = [8]

EPSILON_LIST = [0.05]
ALPHA_LIST = [0.8]

# 設定
MULTI = 1

# プログラム用
ROLE_LEADER = 0
ROLE_MEMBER = 1

def load_parameter(path):
    f = open(path, "r")
    parameter = yaml.full_load(f)
    f.close()

    return parameter
