#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml

# モデル
NUM_PLAYERS = 21
MAX_REP = 21
S = 1
MAX_STEP = 20000
LEADER_SAMPLING_TERM = 100

# カラム
NUM_COLUMN = 22

## 共有
COL_PLAYERID = 0
COL_ANUM = 6
COL_Qa00 = 7
COL_Qa01 = 8
COL_Qa10 = 9
COL_Qa11 = 10
COL_RNUM = 11
COL_QrLEADER = 12
COL_Qr01 = 13
COL_Qr10 = 14
COL_QrMEMBERS = 15
COL_ROLE = 16
COL_Qap00 = 18
COL_Qap01 = 19
COL_Qap10 = 20
COL_Qap11 = 21

## 各報酬
COL_GAME_REWARD = 10
COL_COMUNITY_REWARD = 1
COL_ROLE_REWARD = 17

## 所属コミュニティー
COL_STAYING_COMUNITY = 21

## 成員用
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
MULTI = 10

# プログラム用
ROLE_MEMBER = 0
ROLE_LEADER = 1

def load_parameter(path):
    f = open(path, "r")
    parameter = yaml.full_load(f)
    f.close()

    return parameter
