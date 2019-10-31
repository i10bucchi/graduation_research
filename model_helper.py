#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    players[:, COL_APC] = -1
    players[:, COL_APS] = -1
    players[:, COL_ANUM] = -1
    players[:, COL_Qa00] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qa01] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qa10] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qa11] = np.random.rand(NUM_PLAYERS)
    players[:, COL_RNUM] = -1
    players[:, COL_Qr00] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qr01] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qr10] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qr11] = np.random.rand(NUM_PLAYERS)
    players[:, COL_ROLE] = -1

    return players

def get_members_action(members_qa, parameter):
    '''
    abstract:
        epshilon-greedy法により成員の行動を決定する
    input:
        members_qa:    np.array shape=[NUM_MEMBERS, 4]
            全ての成員のQテーブル
        parameter:      dict
            モデルのパラーメター辞書
    output:
        :               np.array shape=[NUM_MEMBERS, 2]
            全ての成員の行動選択
        members_action: np.array shape=[NUM_MEMBERS]
            全ての成員の行動番号
    '''

    members_action = np.argmax(members_qa, axis=1)
    rand = np.random.rand(members_qa.shape[0])
    members_action[rand < parameter['epsilon']] = np.random.randint(0, 4, members_action[rand < parameter['epsilon']].shape[0])
        
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
    rand = np.random.rand()
    
    if rand < parameter['epsilon']:
        leader_action = np.random.randint(0, 4)
    else:
        leader_action = np.argmax(leader_qa)

    return a_l[leader_action], leader_action

def get_members_gain(members_ac, members_as, leader_apc, leader_aps, parameter, num_members=NUM_MEMBERS):
    '''
    abstract:
        各成員と制裁者の行動から成員の利得を算出
    input:
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
        :   np.array shape=[NUM_MEMBERS]
            全ての成員の利得
    '''

    # 社会的ジレンマ下での成員の得点計算
    d = (parameter['cost_cooperate'] * np.sum(members_ac) * parameter['power_social']) / (num_members)
    # 非協力の場合はparameter['cost_cooperate']がもらえる
    cp = (parameter['cost_cooperate'] * (np.ones(num_members) - members_ac))
    # 非支援の場合はparameter['cost_support']がもらえる
    sp = (parameter['cost_support'] * (np.ones(num_members) - members_as))
    # 非協力の場合に制裁者が罰を行使してたら罰される
    pcp = (parameter['punish_size'] * leader_apc * (np.ones(num_members) - members_ac))
    # 非支援の場合に制裁者が罰を行使してたら罰される
    psp = (parameter['punish_size'] * leader_aps * (np.ones(num_members) - members_as))
    # 利得がマイナスまたは0にならないように最小利得を決定
    min_r = parameter['punish_size'] * 2 + 1
    
    return d + cp + sp - pcp - psp + min_r

def get_leaders_gain(members_ac, members_as, leader_apc, leader_aps, parameter, num_members=NUM_MEMBERS):
    '''
    abstract:
        各成員と制裁者の行動から制裁者の利得を算出
    input:
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

    # 制裁者の得点計算
    tax = np.sum(members_as) * parameter['cost_support']
    # 非協力者制裁を行うコストを支払う
    pcc = parameter['cost_punish'] * leader_apc * (np.sum(np.ones(num_members) - members_ac))
    # 非支援者制裁を行うコストを支払う
    psc = parameter['cost_punish'] * leader_aps * (np.sum(np.ones(num_members) - members_as))
    # 利得がマイナスまたは0にならないように最小利得を決定
    min_r = parameter['cost_punish'] * num_members * 2 + 1

    return tax - pcc - psc + min_r

def learning_members(members_qa, rewards, members_anum, parameter):
    '''
    abstract:
        成員の学習を行う
    input:
        members_qa:     np.array shape=[NUM_MEMBERS, 4]
            全ての成員のQテーブル
        rewards:        np.array shape=[NUM_MEMBERS]
            全ての成員の利得
        members_anum:   np.array shape=[NUM_MEMBERS]
            全ての成員の行動番号
    output:
        :   np.array shape=[NUM_MEMBERS, 4]
            全ての成員の更新後のQテーブル
    '''

    # 今回更新するQ値以外のerrorを0にするためのマスク
    mask = np.zeros((NUM_MEMBERS, 4))
    for i, an in enumerate(members_anum):
        mask[i, int(an)] = 1

    # 誤差
    error = mask * (np.tile(rewards,(4,1)).T - members_qa)

    return members_qa + ( parameter['alpha'] * error )

def learning_leader(members_reward, leader_qa, leader_anum, reward, parameter, theta):
    '''
    abstract:
        制裁者の学習を行う
    input:
        members_reward:     np.array shape=[NUM_MEMBERS]
            成員の利得
        leader_qa:          np.array shape=[4]
            制裁者のQテーブル
        leaderanum:         int
            精細者の行動番号
        reward:             int
            制裁者の利得
        parameter:          dict
            実験パラメータ
        theta:              list len=2
            1次ゲームのルール [成員の利益を考慮するか否か, 行動試行期間の差]
    output:
        :   np.array shape=[4]
            更新後のQテーブル
    '''

    # 今回更新するQ値以外のerrorを0にするためのマスク
    mask = np.zeros(4)
    mask[int(leader_anum)] = 1

    if theta[0] == 0 and theta[1] == 0:
        r = reward
    elif theta[0] == 0 and theta[1] == 1:
        r = reward / LEADER_SAMPLING_TERM
    elif theta[0] == 1 and theta[1] == 0:
        r = ( np.mean(members_reward) * reward )
    else:
        r = ( np.mean(members_reward) * reward ) / LEADER_SAMPLING_TERM

    # 誤差
    error = mask * (np.tile(r, 4) - leader_qa)

    return leader_qa + ( parameter['alpha'] * error )

def one_order_game(players, parameter, theta):
    '''
    abstract:
        1次ゲームを決められた回数行いゲーム利得を算出する
    input:
        players:    np.array shape=(NUM_PLAYERS, NUM_COLMUN)
            ゲームプレイヤー
        parameter:  dict
            実験パラメータ
        theta:  list
            1次ゲームのルール
    output:
        players:    np.array shape=(NUM_PLAYERS, NUM_COLUMN)
            ゲームプレイヤー
        :           tuple
            (平均協調率, 平均支援率)
    '''

    # 役割決定
    players[:, COL_ROLE] = ROLE_MEMBER
    players[np.random.randint(0, NUM_PLAYERS), COL_ROLE] = ROLE_LEADER
    members = players[players[:, COL_ROLE] == ROLE_MEMBER, :]
    leader = players[players[:, COL_ROLE] == ROLE_LEADER, :][0]

    # 収集対象データ初期化
    mr_sum = np.zeros(NUM_MEMBERS)
    lr_sum = 0
    c_num = np.zeros(NUM_MEMBERS)
    s_num = np.zeros(NUM_MEMBERS)

    # ゲーム実行
    step = 0
    for i in range(MAX_STEP):
        # 行動決定
        if theta[1] == 0:
            leader[[COL_APC, COL_APS]], leader[COL_ANUM] = get_leader_action(leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], parameter)
        else:
            if i % LEADER_SAMPLING_TERM == 0:
                leader[[COL_APC, COL_APS]], leader[COL_ANUM]  = get_leader_action(leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], parameter)
        members[:, [COL_AC, COL_AS]], members[:, COL_ANUM] = get_members_action(members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], parameter)
        
        # 利得算出
        mrs = get_members_gain(members[:, COL_AC], members[:, COL_AS], leader[COL_APC], leader[COL_APS], parameter)
        lr = get_leaders_gain(members[:, COL_AC], members[:, COL_AS], leader[COL_APC], leader[COL_APS], parameter)
        members[:, COL_P] += mrs
        leader[COL_P] += lr
        
        step += 1

        # 収集対象のデータを記録
        mr_sum += mrs.astype(np.float)
        lr_sum += lr
        c_num += members[:, COL_AC].astype(np.float)
        s_num += members[:, COL_AS].astype(np.float)

        # 学習
        members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = learning_members(
            members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]],
            members[:, COL_P],
            members[:, COL_ANUM],
            parameter
        )
        members[:, COL_P_LOG] += members[:, COL_P]
        members[:, COL_P] = 0

        if theta[1] == 0:
            leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = learning_leader(
                members[:, COL_P_LOG], leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]],
                leader[COL_ANUM],
                leader[COL_P],
                parameter,
                theta
            )
            leader[COL_P] = 0
            members[:, COL_P_LOG] = 0
        else:
            if i % LEADER_SAMPLING_TERM == LEADER_SAMPLING_TERM - 1:
                leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = learning_leader(
                    members[:, COL_P_LOG], leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]],
                    leader[COL_ANUM],
                    leader[COL_P],
                    parameter,
                    theta
                )
                leader[COL_P] = 0
                members[:, COL_P_LOG] = 0
    
    # ゲームの利得を算出
    members[:, COL_RREWARD] = mr_sum / MAX_STEP
    leader[COL_RREWARD] = lr_sum / MAX_STEP

    players[players[:, COL_ROLE] == ROLE_MEMBER, :] = members
    players[players[:, COL_ROLE] == ROLE_LEADER, :] = leader

    return players, ( c_num.sum() / NUM_MEMBERS / MAX_STEP, s_num.sum() / NUM_MEMBERS / MAX_STEP )

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