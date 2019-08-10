#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
from config import *

warnings.filterwarnings('error')

# 行動の対応表
a_l = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

a_l_str = np.array([
    '00',
    '01',
    '10',
    '11'
])

def generate_players():
    '''
    abstract:
        プレイヤーの情報格納データフレームを作成
    input:
        --
    output:
        players: pd.DataFrame NUM_PLAYERS*NUM_COLMUN
            プレイヤーの情報
    '''

    players = pd.DataFrame(columns=[])
    players['point'] = np.zeros(NUM_PLAYERS)
    players['point_log'] = np.zeros(NUM_PLAYERS)
    players['action_c'] = np.full(NUM_PLAYERS, -1)
    players['action_s'] = np.full(NUM_PLAYERS, -1)
    players['action_pc'] = np.full(NUM_PLAYERS, -1)
    players['action_ps'] = np.full(NUM_PLAYERS, -1)
    players['action_number'] = np.full(NUM_PLAYERS, -1)
    players['Qa_00'] = np.random.rand(NUM_PLAYERS)
    players['Qa_01'] = np.random.rand(NUM_PLAYERS)
    players['Qa_10'] = np.random.rand(NUM_PLAYERS)
    players['Qa_11'] = np.random.rand(NUM_PLAYERS)
    players['role'] = ['member' for i in range(NUM_PLAYERS)]
    players.loc[np.random.randint(NUM_PLAYERS), 'role'] = 'leader'

    return players

def get_members_action(members, epsilon=EPSILON):
    '''
    abstract:
        epshilon-greedy法により成員の行動を決定する
    input:
        members:    np.array shape=[NUM_MEMBERS, NUM_COLUMN]
            全てのグループの成員固体
    output:
        members:    np.array shape=[NUM_MEMBERS, NUM_COLUMN]
            全てのグループの成員固体
    '''
    epsilon_c = np.random.rand(members.shape[0])
    epsilon_s = np.random.rand(members.shape[0])
    members_action = np.argmax(members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], axis=1)

    members[:, [COL_AC, COL_AS]] = np.tile(a_l, (members_action.shape[0], 1))[members_action]
    members[:, COL_AC][epsilon_c < epsilon] = ( np.random.rand(members[epsilon_c < epsilon].shape[0]) + 0.5 ).astype(np.int64)
    members[:, COL_AS][epsilon_s < epsilon] = ( np.random.rand(members[epsilon_s < epsilon].shape[0]) + 0.5 ).astype(np.int64)
    for i in range(NUM_MEMBERS):
        members[i, COL_ANUM] = int(np.where(a_l_str == ''.join(members[i, [COL_AC, COL_AS]].astype(np.int64).astype(str).tolist()))[0])
    
    return members

def get_leader_action(leader, epsilon=EPSILON):
    '''
    abstract:
        epshilon-greedy法により制裁者の行動を決定する
    input:
        leader:    np.array shape=[NUM_COLUMN]
            全てのグループの制裁者固体
    output:
        leader:    np.array shape=[NUM_COLUMN]
            全てのグループの制裁者固体
    '''
    epsilon_pa = np.random.rand()
    epsilon_ps = np.random.rand()
    leader_action = np.argmax(leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]])
    leader[[COL_APC, COL_APS]] = a_l[leader_action]
    if epsilon_pa < epsilon:
        leader[COL_APC] = int( np.random.rand() + 0.5 )
    if epsilon_ps < epsilon:
        leader[COL_APS] = int( np.random.rand() + 0.5 )
    leader[COL_ANUM] = int(np.where(a_l_str == ''.join(leader[[COL_APC, COL_APS]].astype(np.int64).astype(str).tolist()))[0])

    # # 挙動テスト用に常に制裁を行うモード
    leader[[COL_APC, COL_APS]] = 1
    leader[COL_ANUM] = 3

    return leader

def get_members_gain(members, leader, parameter):
    '''
    abstract:
        各成員と制裁者の行動から成員の利得を算出
    input:
        members:    np.array shape=[NUM_MEMBERS, NUM_COLUMN]
            成員の役割を持つプレイヤー
        leader:    np.array shape=[NUM_COLUMN]
            制裁者の役割を持つプレイヤー
        parameter:  dict 
            実験パラメータ
    output:
        d + cp + sp - pcp - psp: np.array shape=[NUM_MEMBERS, NUM_COLUMN]
    '''

    # 社会的ジレンマ下での成員の得点計算
    d = (parameter['cost_cooperate'] * np.sum(members[:, COL_AC]) * parameter['power_social']) / (members.shape[0])
    # 非協力の場合はparameter['cost_cooperate']がもらえる
    cp = (parameter['cost_cooperate'] * (np.ones(members.shape[0]) - members[:, COL_AC]))
    # 非支援の場合はparameter['cost_support']がもらえる
    sp = (parameter['cost_support'] * (np.ones(members.shape[0]) - members[:, COL_AS]))
    # 非協力の場合に制裁者が罰を行使してたら罰される
    pcp = (parameter['punish_size'] * leader[COL_APC] * (np.ones(members.shape[0]) - members[:, COL_AC]))
    # 非支援の場合に制裁者が罰を行使してたら罰される
    psp = (parameter['punish_size'] * leader[COL_APS] * (np.ones(members.shape[0]) - members[:, COL_AS]))
    
    return d + cp + sp - pcp - psp

def get_leaders_gain(members, leader, parameter):
    '''
    abstract:
        各成員と制裁者の行動から制裁者の利得を算出
    input:
        members:    np.array shape=[NUM_MEMBERS, NUM_COLUMN]
            成員の役割を持つプレイヤー
        leader:    np.array shape=[NUM_COLUMN]
            制裁者の役割を持つプレイヤー
        parameter:  dict 
            実験パラメータ
    output:
        tax - pcc - psc: np.array shape=[NUM_COLUMN]
    '''

    # 制裁者の得点計算
    tax = np.sum(members[:, COL_AS]) * parameter['cost_support']
    # 非協力者制裁を行うコストを支払う
    pcc = parameter['cost_punish'] * leader[COL_APC] * (np.sum(np.ones(members.shape[0]) - members[:, COL_AC]))
    # 非支援者制裁を行うコストを支払う
    psc = parameter['cost_punish'] * leader[COL_APS] * (np.sum(np.ones(members.shape[0]) - members[:, COL_AS]))

    return tax - pcc - psc

def calc_gain(members, leader, parameter):
    '''
    abstract:
        成員と制裁者の利得を算出しCOL_Pに加算して返す
    input:
        members:    np.array shape=[NUM_MEMBERS-1, NUM_COLUMN]
            成員の役割を持つプレイヤー
        leader:    np.array shape=[NUM_COLUMN]
            制裁者の役割を持つプレイヤー
        parameter:  dict 
            実験パラメータ
    output:
        members:    np.array shape=[NUM_MEMBERS-1, NUM_COLUMN]
            成員の役割を持つプレイヤー
        leader:    np.array shape=[NUM_COLUMN]
            制裁者の役割を持つプレイヤー
    '''

    members[:, COL_P] += get_members_gain(members, leader, parameter)
    leader[COL_P] += get_leaders_gain(members, leader, parameter)

    return members, leader

def softmax_2dim(X):
    '''
    abstract:
        与えられたarrayに対してsoftmax関数を適用する
    input:
        X: np.array
            softmaxの対象となるベクトル
    output:
        expX / sum_expX: np.array
            softmax適用結果
    '''

    expX = np.exp(X)
    sum_expX = np.sum(expX, axis=1)
    sum_expX = np.tile(sum_expX, (X.shape[1],1))
    sum_expX = sum_expX.transpose()
    return expX / sum_expX

def softmax_1dim(X):
    '''
    abstract:
        与えられたarrayに対してsoftmax関数を適用する
    input:
        X: np.array
            softmaxの対象となるベクトル
    output:
        expX / sum_expX: np.array
            softmax適用結果
    '''

    absmax = np.max(np.absolute(X))
    try:
        norm_X = X / absmax
    except RuntimeWarning as exe:
        norm_X = X
    expX = np.exp(norm_X)
    sum_expX = np.sum(expX)
    return expX / sum_expX

def learning_members(members, alpha=ALPHA):
    '''
    abstract:
        成員の学習を行う
    input:
        members:    np.array shape=[NUM_MEMBERS, NUM_COLUMN]
            成員の役割を持つプレイヤー
    output:
        members:    np.array shape=[NUM_MEMBERS, NUM_COLUMN]
            成員の役割を持つプレイヤー
    '''

    # 今回更新するQ値以外のerrorを0にするためのマスク
    mask = np.zeros((members.shape[0], 4))
    for i, an in enumerate(members[:, COL_ANUM]):
        mask[i, int(an)] = 1

    # 更新
    r = members[:, COL_P]
    error = mask * (np.tile(r,(4,1)).T - members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]])
    members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] + ( alpha * error )

    return members

def learning_leader(members, leader, parameter, alpha=ALPHA):
    '''
    abstract:
        制裁者の学習を行う
    input:
        members:    np.array shape=[NUM_MEMBERS, NUM_COLUMN]
            成員の役割を持つプレイヤー
        leader:    np.array shape=[NUM_COLUMN]
            制裁者の役割を持つプレイヤー
        parameter:  dict
            実験パラメータ
    output:
        leader: np.array shape=[NUM_GROUPS, NUM_COLUMN]
            制裁者の役割を持つプレイヤー
    '''

    # 今回更新するQ値以外のerrorを0にするためのマスク
    mask = np.zeros(4)
    mask[int(leader[COL_ANUM])] = 1

    # マイナスの掛け算にならないような処置
    # r = ( np.sum(members[:, COL_P_LOG]) + (2*parameter['punish_size']) ) * ( leader[COL_P] + (NUM_MEMBERS*2*parameter['cost_punish']) )
    r = np.mean(members[:, COL_P_LOG]) + leader[COL_P]
    # 更新
    error = mask * (np.tile(r, 4) - leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]])
    leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] + ( alpha * error )
    return leader