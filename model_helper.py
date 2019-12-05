#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import warnings
import copy
import pandas as pd
from tqdm import tqdm
from config import *

warnings.filterwarnings('error')

def generate_players():
    '''
    abstract:
        プレイヤーの情報格納NumpyArrayを作成
    input:
        --
    output:
        players: np.array shape=(NUM_PLAYERS, NUM_COLMUN)
            プレイヤーの情報
    '''

    players = np.zeros((NUM_PLAYERS, NUM_COLUMN))
    players[:, COL_AC] = -1
    players[:, COL_AS] = -1
    players[:, COL_ANON] = -1
    players[:, COL_APC] = -1
    players[:, COL_APS] = -1
    players[:, COL_ANUM] = -1
    players[:, COL_Qa00] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qa01] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qa10] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qa11] = np.random.rand(NUM_PLAYERS)
    players[:, COL_QaNON] = np.random.rand(NUM_PLAYERS)
    players[:, COL_ROLE] = -1

    return players

def get_members_action(members_qa, parameter):
    '''
    abstract:
        epshilon-greedy法により成員の行動を決定する
    input:
        members_qa:    np.array shape=[NUM_MEMBERS, 5]
            全ての成員のQテーブル
        parameter:      dict
            モデルのパラーメター辞書
    output:
        :               np.array shape=[NUM_MEMBERS, 2]
            全ての成員の行動選択
        members_action: np.array shape=[NUM_MEMBERS]
            全ての成員の行動番号
    '''

    a_l = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])

    members_action = np.argmax(members_qa, axis=1)
    rand = np.random.rand(members_qa.shape[0])
    members_action[rand < parameter['epsilon']] = np.random.randint(0, 5, members_action[rand < parameter['epsilon']].shape[0])
        
    return np.tile(a_l, (members_action.shape[0], 1))[members_action], members_action

def get_leader_action(leader_qa, parameter):
    '''
    abstract:
        epshilon-greedy法により制裁者の行動を決定する
    input:
        leader_qa:    np.array shape=[4]
            制裁者のQテーブル
        parameter:      dict
            モデルのパラーメター辞書
    output:
        :               np.array shape=[2]
            制裁者の行動選択
        leader_action:  int
            制裁者の行動番号
    '''

    a_l = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    rand = np.random.rand()
    
    if rand < parameter['epsilon']:
        leader_action = np.random.randint(0, 4)
    else:
        leader_action = np.argmax(leader_qa)

    return a_l[leader_action], leader_action

def get_members_gain(members_non, members_ac, members_as, leader_apc, leader_aps, parameter, num_members=NUM_MEMBERS):
    '''
    abstract:
        各成員と制裁者の行動から成員の利得を算出
    input:
        members_non:   np.array shape=[NUM_MEMBERS]
            成員の公共財ゲーム参加費参加の意思
        members_ac:    np.array shape=[NUM_MEMBERS]
            成員の協調の選択の有無
        members_as:    np.array shape=[NUM_MEMBERS]
            成員の支援の選択の有無
        leader_apc:    int
            制裁者の非協調者制裁の選択の有無
        leader_aps:    int
            制裁者の非支援者制裁の選択の有無
        parameter:      dict 
            実験パラメータ
    output:
        r:   np.array shape=[NUM_MEMBERS]
            全ての成員の利得
    '''

    # 非参加者の得点
    income = parameter['cost_cooperate'] * parameter['power_social'] / 2

    # 参加者が2名以上から公共財ゲームが可能
    if np.sum(members_non) >= 2:
        # 社会的ジレンマ下での成員の得点計算
        d = (parameter['cost_cooperate'] * np.sum(members_ac) * parameter['power_social']) / (np.sum(members_non))
        # 非協力の場合はparameter['cost_cooperate']がもらえる
        cp = (parameter['cost_cooperate'] * (np.ones(num_members) - members_ac))
        # 非支援の場合はparameter['cost_support']がもらえる
        sp = (parameter['cost_support'] * (np.ones(num_members) - members_as))
        # 非協力の場合に制裁者が罰を行使してたら罰される
        pcp = (parameter['punish_size'] * leader_apc * (np.ones(num_members) - members_ac))
        # 非支援の場合に制裁者が罰を行使してたら罰される
        psp = (parameter['punish_size'] * leader_aps * (np.ones(num_members) - members_as))
        r_pgg = members_non * ( d + cp + sp - pcp - psp )
    else:
        r_pgg = np.full(NUM_MEMBERS, income)

    
    r = np.full(NUM_MEMBERS, income)
    r[members_non == 0] = income
    r[members_non == 1] = r_pgg[members_non == 1]
    
    return r

def get_leaders_gain(members_non, members_ac, members_as, leader_apc, leader_aps, parameter, num_members=NUM_MEMBERS):
    '''
    abstract:
        各成員と制裁者の行動から制裁者の利得を算出
    input:
        members_non:   np.array shape=[NUM_MEMBERS]
            成員の公共財ゲーム参加費参加の意思
        members_ac:    np.array shape=[NUM_MEMBERS]
            成員の協調の選択の有無
        members_as:    np.array shape=[NUM_MEMBERS]
            成員の支援の選択の有無
        leader_apc:    int
            制裁者の非協調者制裁の選択の有無
        leader_aps:    int
            制裁者の非支援者制裁の選択の有無
        parameter:  dict 
            実験パラメータ
    output:
        :   int
            制裁者の利得
    '''

    # 非参加者数を定義
    num_non = NUM_MEMBERS - np.sum(members_non)
    # 制裁者の得点計算
    tax = np.sum(members_as) * parameter['cost_support']
    # 非協力者制裁を行うコストを支払う(非参加者は罰しない)
    pcc = parameter['cost_punish'] * leader_apc * (np.sum(np.ones(num_members) - members_ac) - num_non)
    # 非支援者制裁を行うコストを支払う(非参加者は罰しない)
    psc = parameter['cost_punish'] * leader_aps * (np.sum(np.ones(num_members) - members_as) - num_non)

    return tax - pcc - psc

def learning_members(members_qa, rewards, members_anum, parameter):
    '''
    abstract:
        成員の学習を行う
    input:
        members_qa:     np.array shape=[NUM_MEMBERS, 5]
            全ての成員のQテーブル
        rewards:        np.array shape=[NUM_MEMBERS]
            全ての成員の利得
        members_anum:   np.array shape=[NUM_MEMBERS]
            全ての成員の行動番号
    output:
        :   np.array shape=[NUM_MEMBERS, 5]
            全ての成員の更新後のQテーブル
    '''

    # 今回更新するQ値以外のerrorを0にするためのマスク
    mask = np.zeros((NUM_MEMBERS, 5))
    for i, an in enumerate(members_anum):
        mask[i, int(an)] = 1

    # 誤差
    error = mask * (np.tile(rewards,(5,1)).T - members_qa)

    return members_qa + ( parameter['alpha'] * error )

def learning_leader(leader_qa, leader_anum, reward, parameter):
    '''
    abstract:
        制裁者の学習を行う
    input:
        leader_qa:          np.array shape=[4]
            制裁者のQテーブル
        leaderanum:         int
            精細者の行動番号
        reward:             int
            制裁者の利得
        parameter:          dict
            実験パラメータ
    output:
        :   np.array shape=[4]
            更新後のQテーブル
    '''

    # 今回更新するQ値以外のerrorを0にするためのマスク
    mask = np.zeros(4)
    mask[int(leader_anum)] = 1
    
    r = reward / LEADER_SAMPLING_TERM

    # 誤差
    error = mask * (np.tile(r, 4) - leader_qa)

    return leader_qa + ( parameter['alpha'] * error )

def exec_game(players, parameter):
    '''
    abstract:
        1次ゲームを決められた回数行いゲーム利得を算出する
    input:
        players:    np.array shape=(NUM_PLAYERS, NUM_COLMUN)
            ゲームプレイヤー
        parameter:  dict
            実験パラメータ
    output:
        players:    np.array shape=(NUM_PLAYERS, NUM_COLUMN)
            ゲームプレイヤー
        df:         pd.DataFrame columns=['N', 'C', 'D', 'S', 'nS'] length=MAX_STEP
            各行動を行ったゲームステップごとの人数を格納
    '''

    # 役割決定
    players[:, COL_ROLE] = ROLE_MEMBER
    players[np.random.randint(0, NUM_PLAYERS), COL_ROLE] = ROLE_LEADER
    members = players[players[:, COL_ROLE] == ROLE_MEMBER, :]
    leader = players[players[:, COL_ROLE] == ROLE_LEADER, :][0]

    # 収集対象データ初期化
    n_num_list = []
    c_num_list = []
    d_num_list = []
    s_num_list = []
    ns_num_list = []

    # ゲーム実行
    step = 0
    for i in tqdm(range(MAX_STEP)):
        # 行動決定
        if i % LEADER_SAMPLING_TERM == 0:
            leader[[COL_APC, COL_APS]], leader[COL_ANUM]  = get_leader_action(leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], parameter)
        members[:, [COL_ANON, COL_AC, COL_AS]], members[:, COL_ANUM] = get_members_action(members[:, [COL_QaNON, COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], parameter)
        
        # 利得算出
        mrs = get_members_gain(members[:, COL_ANON], members[:, COL_AC], members[:, COL_AS], leader[COL_APC], leader[COL_APS], parameter)
        lr = get_leaders_gain(members[:, COL_ANON], members[:, COL_AC], members[:, COL_AS], leader[COL_APC], leader[COL_APS], parameter)
        members[:, COL_P] += mrs
        leader[COL_P] += lr
        
        step += 1

        # 収集対象のデータを記録
        n_num_list.append( NUM_MEMBERS - np.sum(members[:, COL_ANON]))
        c_num_list.append(np.sum(members[:, COL_AC]))
        d_num_list.append( np.sum(members[:, COL_ANON]) - np.sum(members[:, COL_AC]) )
        s_num_list.append(np.sum(members[:, COL_AS]))
        ns_num_list.append( np.sum(members[:, COL_ANON]) - np.sum(members[:, COL_AS]) )

        # 学習
        members[:, [COL_QaNON, COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = learning_members(
            members[:, [COL_QaNON, COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]],
            members[:, COL_P],
            members[:, COL_ANUM],
            parameter
        )
        members[:, COL_P] = 0

        if i % LEADER_SAMPLING_TERM == LEADER_SAMPLING_TERM - 1:
            leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = learning_leader(
                leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]],
                leader[COL_ANUM],
                leader[COL_P],
                parameter,
            )
            leader[COL_P] = 0

    players[players[:, COL_ROLE] == ROLE_MEMBER, :] = members
    players[players[:, COL_ROLE] == ROLE_LEADER, :] = leader

    df = pd.DataFrame()
    df['step'] = range(MAX_STEP)
    df['N'] = n_num_list
    df['C'] = c_num_list
    df['D'] = d_num_list
    df['S'] = s_num_list
    df['nS'] = ns_num_list

    return players, df.astype(np.int64)