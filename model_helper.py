#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
import copy
from tqdm import tqdm
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
    players['rule_number'] = np.full(NUM_PLAYERS, -1)
    players['Qr_00'] = np.random.rand(NUM_PLAYERS)
    players['Qr_01'] = np.random.rand(NUM_PLAYERS)
    players['Qr_10'] = np.random.rand(NUM_PLAYERS)
    players['Qr_11'] = np.random.rand(NUM_PLAYERS)
    players['role'] = ['undefined' for i in range(NUM_PLAYERS)]
    players['rule_reward'] = np.zeros(NUM_PLAYERS)

    return players

def get_members_action(members, parameter):
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

    members_action = np.argmax(members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], axis=1)
    rand = np.random.rand(members.shape[0])
    members_action[rand < parameter['epsilon']] = np.random.randint(0, 4, members_action[rand < parameter['epsilon']].shape[0])

    members[:, [COL_AC, COL_AS]] = np.tile(a_l, (members_action.shape[0], 1))[members_action]
    members[:, COL_ANUM] = members_action
        
    return members

def get_leader_action(leader, parameter):
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
    rand = np.random.rand()
    
    if rand < parameter['epsilon']:
        leader_action = np.random.randint(0, 4)
    else:
        leader_action = np.argmax(leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]])
    
    leader[[COL_APC, COL_APS]] = a_l[leader_action]
    leader[COL_ANUM] = leader_action

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
    # 利得がマイナスまたは0にならないように最小利得を決定
    min_r = parameter['punish_size'] * 2 + 1
    
    return d + cp + sp - pcp - psp + min_r

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
    # 利得がマイナスまたは0にならないように最小利得を決定
    min_r = parameter['cost_punish'] * members.shape[0] * 2 + 1

    return tax - pcc - psc + min_r

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

    mrs = get_members_gain(members, leader, parameter)
    lr = get_leaders_gain(members, leader, parameter)
    members[:, COL_P] += mrs
    leader[COL_P] += lr

    return members, leader, mrs, lr

def learning_members(members, parameter):
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
    members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] + ( parameter['alpha'] * error )

    return members

def learning_leader(members, leader, parameter, theta):
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
        theta: list len=2
            1次ゲームのルール [成員の利益を考慮するか否か, 行動試行期間の差]
    output:
        leader: np.array shape=[NUM_GROUPS, NUM_COLUMN]
            制裁者の役割を持つプレイヤー
    '''

    # 今回更新するQ値以外のerrorを0にするためのマスク
    mask = np.zeros(4)
    mask[int(leader[COL_ANUM])] = 1

    if theta[0] == 0 and theta[1] == 0:
        r = leader[COL_P]
    elif theta[0] == 0 and theta[1] == 1:
        r = leader[COL_P] / LEADER_SAMPLING_TERM
    elif theta[0] == 1 and theta[1] == 0:
        r = ( np.mean(members[:, COL_P_LOG]) * leader[COL_P] )
    else:
        r = ( np.mean(members[:, COL_P_LOG]) * leader[COL_P] ) / LEADER_SAMPLING_TERM

    # 更新
    error = mask * (np.tile(r, 4) - leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]])
    leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] + ( parameter['alpha'] * error )
    return leader

def one_order_game(players, parameter, theta):
    '''
    abstract:
        1次ゲームを決められた回数行いゲーム利得を算出する
    input:
        players:    pd.DataFrame [NUM_PLAYERS, NUM_COLUMN]
            成員の役割を持つプレイヤー
        parameter:  dict
            実験パラメータ
        theta:  list
            1次ゲームのルール
    output:
    '''

    players['role'] = 'member'
    players.loc[np.random.randint(0, NUM_PLAYERS), 'role'] = 'leader'
    members = players[players['role'] == 'member'].values
    leader = players[players['role'] == 'leader'].values[0]

    mr_l = []
    lr_l = []

    step = 0
    for i in range(MAX_STEP):
        # ゲーム
        if theta[1] == 0:
            leader = get_leader_action(leader, parameter)
        else:
            if i % LEADER_SAMPLING_TERM == 0:
                leader = get_leader_action(leader, parameter)
        members = get_members_action(members, parameter)
        members, leader, mrs, lr = calc_gain(members, leader, parameter)
        step += 1

        mr_l.append(mrs)
        lr_l.append(lr)

        # 学習
        members = learning_members(members, parameter)
        members[:, COL_P_LOG] += members[:, COL_P]
        members[:, COL_P] = 0
        if theta[1] == 0:
            leader = learning_leader(members, leader, parameter, theta)
            leader[COL_P] = 0
            members[:, COL_P_LOG] = 0
        else:
            if i % LEADER_SAMPLING_TERM == LEADER_SAMPLING_TERM - 1:
                leader = learning_leader(members, leader, parameter, theta)
                leader[COL_P] = 0
                members[:, COL_P_LOG] = 0
    
    # ゲームの利得を算出
    members[:, COL_RREWARD] = np.mean(mr_l, axis=0)
    leader[COL_RREWARD] = np.mean(lr_l)

    players[players['role'] == 'member'] = members
    players[players['role'] == 'leader'] = leader

    return players

def get_players_rule(players, epshilon=0.5):
    '''
    abstract:
        epshilon-greedy法によりプレイヤーの支持するゲームルールを決定する
    input:
        players:    np.array shape=[NUM_PLAYERS, NUM_COLUMN]
            全てのグループの成員固体
    output:
        players_rule:   np.array shape=[NUM_PLAYERS]
            各プレイヤーが選択したゲームルール配列
    '''

    players_rule = np.argmax(players[:, [COL_Qr00, COL_Qr01, COL_Qr10, COL_Qr11]], axis=1)
    rand = np.random.rand(players.shape[0])
    players_rule[rand < epshilon] = np.random.randint(0, 4, players_rule[rand < epshilon].shape[0])

    return players_rule

def get_gaming_rule(players):
    '''
    abstract:
        各プレイヤーが支持するルールで多数決を行う
    input:
        players:    np.array shape=[NUM_PLAYERS, NUM_COLUMN]
            全てのグループの成員固体
    output:
        :    int
            多数決で決まったルールの番号(最大表が2つ以上あった場合は-1を返す)
    '''

    rule_hyonum = np.bincount(players[:, COL_RNUM].astype(np.int64))
    max_rule = np.argmax(rule_hyonum)

    if np.sum(rule_hyonum[max_rule] == rule_hyonum) == 1:
        return max_rule
    else:
        return -1

def get_rule_gain(players):
    '''
    abstract:
        Qaの最大値を評価とする
    input:
        players:    np.array shape=[NUM_PLAYERS, NUM_COLUMN]
            全てのグループの成員固体
    output:
        :    np.array shape=[NUM_PLAYERS]
            多数決で決まったルールの番号(最大表が2つ以上あった場合は-1を返す)
    '''
    return np.max(players[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], axis=1)

def learning_rule(players, rule_number, alpha=0.8):
    '''
    abstract:
        ゲームルールの学習を行う
    input:
        players:    np.array shape=[NUM_PLAYERS, NUM_COLUMN]
            成員の役割を持つプレイヤー
        rule_number:    int
            採用したルール番号
        alpha:          float
            学習率
    output:
        :    np.array shape=[NUM_PLAYERS, 4]
            更新したQ値
    '''

    # 各プレイヤーにとってのルールの評価値
    r = players[:, COL_RREWARD]

    # 今回更新するQ値以外のerrorを0にするためのマスク
    mask = np.zeros((players.shape[0], 4))
    mask[:, rule_number] = 1

    # 更新
    error = mask * (np.tile(r,(4,1)).T - players[:, [COL_Qr00, COL_Qr01, COL_Qr10, COL_Qr11]])
    return players[:, [COL_Qr00, COL_Qr01, COL_Qr10, COL_Qr11]] + ( alpha * error )