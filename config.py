#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml

# モデル
NUM_MEMBERS = 20
NUM_GROUPS = 20
NUM_CANDIDATES = 5
MAX_REP = 51
S = 1
MAX_GENERATION = 500000
MAX_GAME = 10
MAX_SIMU = 100
MAX_TERM_OF_OFFICE = 10000
FREQ_EVOL_LEADERS = 100

# カラム
NUM_COLUMN = 10

## 共有
COL_P = 0

## 成員用
COL_P_LOG = 1
COL_AC = 2
COL_AS = 3
COL_GC = 6
COL_GS = 7

## 制裁者用
COL_APC = 4
COL_APS = 5
COL_GPC = 8
COL_GPS = 9

# パラメータ
COST_C_LIST = [4]
COST_S_LIST = [2]
COST_P_LIST = [2]
SP_LIST = [4]
PUNISH_SIZE_LIST = [8]

PROB_EVOL_IN_GROUP = 0.9
PROB_MUTATION = 0.005

# 設定
MULTI = 10

def load_parameter(path):
    f = open(path, "r")
    parameter = yaml.load(f)
    f.close()

    return parameter
