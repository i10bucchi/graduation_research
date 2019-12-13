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
    players[:, COL_PLAYERID] = range(NUM_PLAYERS)
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
    players[:, COL_QrLEADER] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qr01] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qr10] = np.random.rand(NUM_PLAYERS)
    players[:, COL_QrMEMBERS] = np.random.rand(NUM_PLAYERS)
    players[:, COL_ROLE] = -1
    players[:, COL_Qap00] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qap01] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qap10] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qap11] = np.random.rand(NUM_PLAYERS)

    return players

def get_action_inpgg(qa, parameter):
    '''
    abstract:
        epshilon-greedy法によりプレイヤーの行動を決定する
    input:
        qa:    np.array shape=[-1, 4]
            全ての成員のQテーブル
        parameter:      dict
            モデルのパラーメター辞書
    output:
        :               np.array shape=[-1, 2]
            全ての成員の行動選択
        action: np.array shape=[-1,]
            全ての成員の行動番号
    '''

    action = np.argmax(qa, axis=1)
    rand = np.random.rand(qa.shape[0])
    action[rand < parameter['epsilon']] = np.random.randint(0, 4, action[rand < parameter['epsilon']].shape[0])
        
    return np.tile(a_l, (action.shape[0], 1))[action], action

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

def learning_action(qa, rewards, anum, alpha=0.8):
    '''
    abstract:
        成員の学習を行う
    input:
        qa:     np.array shape=[-1, 4]
            全ての成員のQテーブル
        rewards:        np.array shape=[-1,]
            全ての成員の利得
        anum:   np.array shape=[-1,]
            全ての成員の行動番号
    output:
        :   np.array shape=[-1,]
            全ての成員の更新後のQテーブル
    '''

    # 今回更新するQ値以外のerrorを0にするためのマスク
    mask = np.zeros_like(qa)
    for i, an in enumerate(anum):
        mask[i, int(an)] = 1

    # 誤差
    error = mask * (np.tile(rewards,(4,1)).T - qa)

    return qa + ( alpha * error )

def get_newcomunity(members_comunity_reward, members_staying_comunity):
    '''
    abstract:
        どこのコミュニティに加入するかを決定する
    input:
        members_comunity_reward:    np.array shape=[-1,]
            自身が現在のコミュニティーで得ることのできた利得
        members_staying_comunity:   np.array shape=[-1,]
            各成員が所属しているコミュニティの制裁者のPLAYERID
    output:
        :   np.array shape=[-1,]
            各成員が次に所属するコミュニティの制裁者のPLAYERID(無所属の場合は-1)
    '''

    
    

def exec_pgg(players, parameter):
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

    members = players[players[:, COL_ROLE] == ROLE_MEMBER, :]
    leaders = players[players[:, COL_ROLE] == ROLE_LEADER, :]

    # ゲーム実行
    for i in range(20):
        # コミュニティの模倣(移動)
        members[:, COL_COMUNITY] = get_newcomunity(members[:, COL_COMUNITY_REWARD])
        members[:, COL_COMUNITY_REWARD] = 0
        
        # コミュニティないで罰ありPGG
        for i in range(MAX_STEP):
            # 行動決定
            if i % LEADER_SAMPLING_TERM == 0:
                leaders[:, [COL_APC, COL_APS]], leaders[COL_ANUM]  = get_action_inpgg(leaders[:, [COL_Qap00, COL_Qap01, COL_Qap10, COL_Qap11]], parameter)
            members[:, [COL_AC, COL_AS]], members[:, COL_ANUM] = get_action_inpgg(members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], parameter)
            
            # 利得算出
            mrs = get_members_gain(members[:, COL_AC], members[:, COL_AS], leaders[COL_APC], leaders[COL_APS], parameter)
            lrs = get_leaders_gain(members[:, COL_AC], members[:, COL_AS], leaders[COL_APC], leaders[COL_APS], parameter)
            members[:, COL_GAME_REWARD] = mrs
            leaders[:, COL_GAME_REWARD] = lrs
            members[:, COL_COMUNITY_REWARD] += mrs
            members[:, COL_ROLE_REWARD] += mrs
            leaders[:, COL_ROLE_REWARD] += lrs

            # 学習
            members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = learning_action(
                members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]],
                members[:, COL_GAME_REWARD],
                members[:, COL_ANUM],
                alpha=parameter['alpha']
            )
            members[:, COL_GAME_REWARD] = 0

            if i % LEADER_SAMPLING_TERM == LEADER_SAMPLING_TERM - 1:
                leaders[[COL_Qap00, COL_Qap01, COL_Qap10, COL_Qap11]] = learning_action(
                    leaders[:, [COL_Qap00, COL_Qap01, COL_Qap10, COL_Qap11]],
                    leaders[:, COL_GAME_REWARD],
                    leaders[:, COL_ANUM],
                    alpha=parameter['alpha']
                )
                leaders[COL_GAME_REWARD] = 0

    players[players[:, COL_ROLE] == ROLE_MEMBER, :] = members
    players[players[:, COL_ROLE] == ROLE_LEADER, :] = leaders

    return players

def get_players_role(players_qr, epshilon=0.1):
    '''
    abstract:
        epshilon-greedy法により次のゲームでのプレイヤーの役割を決定する
    input:
        players:    np.array shape=[NUM_PLAYERS, 2(COL_QrLEADER, COL_QrMEMBERS)]
            全てのグループの成員固体
    output:
        players_rule:   np.array shape=[NUM_PLAYERS]
            各プレイヤーが選択したゲームルール配列
    '''

    players_role = np.argmax(players_qr, axis=1)
    rand = np.random.rand(players_qr.shape[0])
    players_role[rand < epshilon] = np.random.randint(0, 4, players_qr[rand < epshilon].shape[0])

    return players_role

def learning_role(players_qr, rewards, players_role, alpha=0.8):
    '''
    abstract:
        ゲームルールの学習を行う
    input:
        players_qr:    np.array shape=[NUM_PLAYERS, 2]
            プレイヤーのルールQテーブル
        rewards:
            ルールの報酬
        rule_number:    int
            採用したルール番号
        alpha:          float
            学習率
    output:
        :    np.array shape=[NUM_PLAYERS, 2]
            更新したQ値
    '''

    # 今回更新するQ値以外のerrorを0にするためのマスク
    mask = np.zeros((NUM_PLAYERS, 2))
    mask[players_role == ROLE_LEADER, 0] = 1
    mask[players_role == ROLE_LEADER, 1] = 1

    # 更新
    error = mask * (np.tile(rewards,(2,1)).T - players_qr)

    return players_qr + ( alpha * error )