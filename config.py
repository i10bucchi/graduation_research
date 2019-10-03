#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml

# モデル
NUM_PLAYERS = 21
NUM_MEMBERS = NUM_PLAYERS-1
MAX_REP = 5
S = 1
MAX_STEP = 2000
LEADER_SAMPLING_TERM = 100

# カラム
NUM_COLUMN = 18

## 共有
COL_P = 0
COL_ANUM = 6
COL_Qa00 = 7 # 成員ならQ(c, s), 制裁者ならQ(pc, ps)
COL_Qa01 = 8
COL_Qa10 = 9
COL_Qa11 = 10
COL_RNUM = 11
COL_Qr00 = 12
COL_Qr01 = 13
COL_Qr10 = 14
COL_Qr11 = 15
COL_ROLE = 16
COL_RREWARD = 17

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
SP_LIST = [4]
PUNISH_SIZE_LIST = [8]

EPSILON_LIST = [0.05]
ALPHA_LIST = [0.8]

# 設定
MULTI = 4

def load_parameter(path):
    f = open(path, "r")
    parameter = yaml.full_load(f)
    f.close()

    return parameter
