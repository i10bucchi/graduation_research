#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml

# モデル
NUM_PLAYERS = 21
NUM_MEMBERS = NUM_PLAYERS-1
MAX_REP = 12
S = 1
MAX_STEP = 10000
LEADER_SAMPLING_TERM = 100

# カラム
NUM_COLUMN = 12

## 共有
COL_P = 0
COL_ANUM = 6
COL_Qa00 = 7 # 成員ならQ(c, s), 制裁者ならQ(pc, ps)
COL_Qa01 = 8
COL_Qa10 = 9
COL_Qa11 = 10
COL_ROLE = 11

## 成員用
COL_P_LOG = 1
COL_AC = 2
COL_AS = 3

## 制裁者用
COL_APC = 4
COL_APS = 5

# パラメータ
COST_C_LIST = [4]
COST_S_LIST = [2]
COST_P_LIST = [2]
SP_LIST = [16]
PUNISH_SIZE_LIST = [16]

EPSILON_LIST = [0.2]
ALPHA_LIST = [0.4]

# 設定
MULTI = 4

def load_parameter(path):
    f = open(path, "r")
    parameter = yaml.full_load(f)
    f.close()

    return parameter
