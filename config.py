#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml

# モデル
NUM_PLAYERS = 21
NUM_MEMBERS = NUM_PLAYERS-1
MAX_REP = 13
S = 1
MAX_STEP = 200000
LEADER_SAMPLING_TERM = 100

# カラム
NUM_COLUMN = 14

## 共有
COL_P = 0
COL_ANUM = 6
COL_QaNON = 7
COL_Qa00 = 8 # 成員ならQ(c, s), 制裁者ならQ(pc, ps)
COL_Qa01 = 9
COL_Qa10 = 10
COL_Qa11 = 11
COL_ROLE = 12

## 成員用
COL_AC = 1
COL_AS = 2
COL_ANON = 3

## 制裁者用
COL_APC = 4
COL_APS = 5

# パラメータ
COST_C_LIST = [4]
COST_S_LIST = [2]
COST_P_LIST = [2]
SP_LIST = [4]
PUNISH_SIZE_LIST = [8]

EPSILON_LIST = [0.05]
ALPHA_LIST = [0.8]

# 設定
MULTI = 4

# プログラム用
ROLE_MEMBER = 1
ROLE_LEADER = 2

def load_parameter(path):
    f = open(path, "r")
    parameter = yaml.full_load(f)
    f.close()

    return parameter
