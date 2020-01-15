#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml

# モデル
NUM_PLAYERS = 21
NUM_MEMBERS = NUM_PLAYERS-1
MAX_REP = 2
S = 1
MAX_STEP = 100000
LEADER_SAMPLING_TERM = 100

# カラム
NUM_COLUMN = 33

## 共有
COL_P = 0
COL_ANUM = 7
COL_Qa000 = 8 # Q(c, s, f)
COL_Qa001 = 9
COL_Qa010 = 10
COL_Qa011 = 11
COL_Qa100 = 12 # Q(c, s, f)
COL_Qa101 = 13
COL_Qa110 = 14
COL_Qa111 = 15 
COL_RNUM = 16
COL_Qr00 = 17
COL_Qr01 = 18
COL_Qr10 = 19
COL_Qr11 = 24
COL_ROLE = 25
COL_RREWARD = 26
COL_Qap00 = 27 # Q(pc, ps)
COL_Qap01 = 28
COL_Qap10 = 29
COL_Qap11 = 30

## 成員用
COL_P_LOG = 1
COL_AC = 2
COL_AS = 3
COL_AF = 4
COL_P_F = 31
COL_F_TIMER = 32

## 制裁者用
COL_APC = 5
COL_APS = 6

# パラメータ
COST_C_LIST = [4]
COST_S_LIST = [2]
COST_P_LIST = [2]
SP_LIST = [4]
PUNISH_SIZE_LIST = [8]

EPSILON_LIST = [0.05]
ALPHA_LIST = [0.8]

# 設定
MULTI = 10

# プログラム用
ROLE_MEMBER = 1
ROLE_LEADER = 2

def load_parameter(path):
    f = open(path, "r")
    parameter = yaml.full_load(f)
    f.close()

    return parameter
