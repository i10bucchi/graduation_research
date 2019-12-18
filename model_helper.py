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
    players[:, COL_AC] = np.nan
    players[:, COL_AS] = np.nan
    players[:, COL_APC] = np.nan
    players[:, COL_APS] = np.nan
    players[:, COL_ANUM] = np.nan
    players[:, COL_Qa00] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qa01] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qa10] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qa11] = np.random.rand(NUM_PLAYERS)
    players[:, COL_QrLEADER] = np.random.rand(NUM_PLAYERS)
    players[:, COL_QrMEMBERS] = np.random.rand(NUM_PLAYERS)
    players[:, COL_ROLE] = np.nan
    players[:, COL_Qap00] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qap01] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qap10] = np.random.rand(NUM_PLAYERS)
    players[:, COL_Qap11] = np.random.rand(NUM_PLAYERS)
    players[:, COL_COMUNITY_REWARD] = 0

    return players

def get_action_inpgg(qa, epsilon=0.02):
    '''
    abstract:
        epsilon-greedy法によりプレイヤーの行動を決定する
    input:
        qa:         np.array shape=[-1, 4]
            全ての成員のQテーブル
        epsilon:   float
            探索的行動をとる確率. default=0.02
    output:
        :               np.array shape=[-1, 2]
            全ての成員の行動選択
        action: np.array shape=[-1,]
            全ての成員の行動番号
    '''

    action = qa.argmax(axis=1)
    rand = np.random.rand(qa.shape[0])
    action[rand < epsilon] = np.random.choice(range(4), size=(rand < epsilon).sum())
    
    return np.tile(a_l, (action.shape[0], 1))[action], action

def get_members_gain(members_ac, members_as, members_cid, leaders_cid, leaders_apc, leaders_aps, parameter):
    '''
    abstract:
        各成員と制裁者の行動から成員の利得を算出
    input:
        members_ac:    np.array shape=[-1,]
            成員の協調の選択の有無
        members_as:    np.array shape=[-1,]
            成員の支援の選択の有無
        members_cid:   np.array shape=[-1,]
            成員が所属しているコミュニティの制裁者のPLAYERID
        leaders_cid:   np.array shape=[-1,]
            制裁者のPLAYERID
        leaders_apc:   np.array shape=[-1,]
            制裁者の非協調者制裁の選択の有無
        leaders_aps:   np.array shape=[-1,]
            制裁者の非支援者制裁の選択の有無
        parameter:      dict 
            実験パラメータ
    output:
        :   np.array shape=[-1,]
            全ての成員の利得
    '''

    # int: 毎ステップの微小な収入
    r_income = parameter['cost_cooperate'] + parameter['cost_support']

    # unique_comunity_id:
    #   np.array shape=[制裁者数(0人のコミュニティを含めない)]: コミュニティID(制裁者のプレイヤーID)
    # num_comunity_members:
    #   np.array shape=[制裁者数(0人のコミュニティを含めない)]: 各コミュニティ内の成員の所属人数
    unique_comunity_id, comunity_size = np.unique(members_cid, return_counts=True)

    # np.array shpae=[成員数]: ユニークなコミュニティID集合をソートした時のインデックス番号
    members_cindex = np.vectorize(lambda x: np.where(unique_comunity_id == x)[0][0])(members_cid)

    # np.array shape=[成員数, 制裁者数(0人のコミュニティを含めない)]: 成員がどのコミュニティに属しているかのone_hotベクトル
    comunity_mask = np.eye(len(unique_comunity_id))[members_cindex]

    # np.array shape=[制裁者数(0人のコミュニティを含めない)]: 各コミュニティ内で協調を選択した人数
    comunity_cooperation = np.dot(members_ac, comunity_mask)
    
    # 公共財ゲームでの各コミュニティーにおける分配得点計算
    d_comunity = parameter['cost_cooperate'] * comunity_cooperation * parameter['power_social'] / comunity_size
    # 公共財ゲームで1人1人が分配される額を算出
    d = np.dot(comunity_mask, d_comunity)
    
    # 協力の場合はparameter['cost_cooperate']が引かれる
    cp = parameter['cost_cooperate'] * members_ac
    
    # 支援の場合はparameter['cost_support']が引かれる
    sp = parameter['cost_cooperate'] * members_as
    
    # np.array shape=[2, 成員数]: 非協調行動と非支援行動へ1をそれ以外の行動は0を持つベクトル
    members_d_ns = np.ones((2, members_ac.shape[0])) - np.array([members_ac, members_as])
    # np.array shape=[2, 制裁者数(0人のコミュニティを含めない)]: 制裁者の制裁の大きさを表すベクトル
    punishment = np.array([
            leaders_apc[np.isin(leaders_cid, unique_comunity_id)],
            leaders_aps[np.isin(leaders_cid, unique_comunity_id)]
        ]) * parameter['punish_size']

    # 非協調の非支援の場合に制裁者が制裁を行使したら罰される
    pp = (comunity_mask * np.dot(members_d_ns.T, punishment)).sum(axis=1)
    
    r = r_income + d - cp - sp - pp
    
    # コミュニティに1人しかいない場合は公共財ゲームが成り立たない
    r[np.dot(comunity_mask, comunity_size) == 1] = r_income

    return r

def get_leaders_gain(members_ac, members_as, members_cid, leaders_cid, leaders_apc, leaders_aps, parameter):
    '''
    abstract:
        各成員と制裁者の行動から制裁者の利得を算出
    input:
        members_ac:    np.array shape=[NUM_MEMBERS]
            成員の協調の選択の有無
        members_as:    np.array shape=[NUM_MEMBERS]
            成員の支援の選択の有無
        leaders_cid:
            制裁者のPLAYERID
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


    # np.array shape=[制裁者数(0人のコミュニティを含めない)]: コミュニティID(制裁者のプレイヤーID)
    unique_comunity_id = np.unique(members_cid)

    # np.array shpae=[成員数]: ユニークなコミュニティID集合をソートした時のインデックス番号
    members_cindex = np.vectorize(lambda x: np.where(unique_comunity_id == x)[0][0])(members_cid)

    # np.array shape=[成員数, 制裁者数(0人のコミュニティを含めない)]: 成員がどのコミュニティに属しているかのone_hotベクトル
    comunity_mask = np.eye(len(unique_comunity_id))[members_cindex]

    # 税金の徴収
    tax = np.dot(members_as, comunity_mask) * parameter['cost_support']

    # np.array shape=[2, 成員数]: 非協調行動と非支援行動へ1をそれ以外の行動は0を持つベクトル
    members_d_ns = np.ones((2, members_ac.shape[0])) - np.array([members_ac, members_as])
    # np.array shape=[2, 制裁者数(0人のコミュニティを含めない)]: 各コミュニティの非協調者, 非支援者人数
    comunity_d_ns_size = np.dot(members_d_ns, comunity_mask)
    
    # 制裁費用の算出
    p_cost = np.diag(
        np.dot(
            np.array([
                leaders_apc[np.isin(leaders_cid, unique_comunity_id)],
                leaders_aps[np.isin(leaders_cid, unique_comunity_id)]
            ]).T,
            comunity_d_ns_size
        )
    )

    r = np.zeros(leaders_cid.shape[0])
    r[np.isin(leaders_cid, unique_comunity_id)] = tax - p_cost

    return r

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
    for i in range(4):
        mask[anum == i, i] = 1
    # for i, an in enumerate(anum):
    #     mask[i, int(an)] = 1

    # 誤差
    error = mask * (np.tile(rewards,(4,1)).T - qa)

    return qa + ( alpha * error )

def get_newcomunity(comunity_reward, comunity_ids, num_members, mu=0.03):
    '''
    abstract:
        どこのコミュニティに加入するかを決定する
    input:
        comunity_reward:    np.array shape=[-1,]
            各コミュニティー内で成員が得ることのできた利得
        comunity_ids:       np.array shape=[-1,]
            各コミュニティの制裁者のPLAYERID
        num_members:        int
            成員の人数
        mu:                 float
            探索的行動をとる確率. default=0.05
    output:
        :   np.array shape=[-1,]
            各成員が次に所属するコミュニティの制裁者のPLAYERID(無所属の場合は-1)
    '''
    cr = comunity_reward / COMUNITY_MOVE_TERM
    if np.sum(cr < 0) > 0:
        cr += -1 * cr.min() + 1
    
    cr = np.power(cr, 4)
    prob = cr / cr.sum()
    next_comunity = np.random.choice(comunity_ids, size=num_members, p=prob)
    rand = np.random.rand(num_members)
    next_comunity[rand < mu] = np.random.choice(comunity_ids, size=(rand < mu).sum())
    return next_comunity

    # next_comunity = np.zeros(num_members)
    # next_comunity[:] = comunity_ids[np.argmax(comunity_reward)]
    # rand = np.random.rand(num_members)
    # next_comunity[rand < mu] = np.random.choice(comunity_ids, size=(rand < mu).sum())
    # return next_comunity

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

    # qa_histry = np.zeros((MAX_STEP, 4))
    # qap_histry = np.zeros((MAX_STEP, 4))
    # comu_n_histry = np.zeros((int(MAX_STEP/COMUNITY_MOVE_TERM), NUM_PLAYERS))
    # comu_r_histry = np.zeros((int(MAX_STEP/COMUNITY_MOVE_TERM), NUM_PLAYERS))

    if np.sum(np.isnan(leaders[:, COL_COMUNITY_REWARD])) > 0:
        # 情報なしコミュニティの評価値
        leaders[np.isnan(leaders[:, COL_COMUNITY_REWARD]), COL_COMUNITY_REWARD] = leaders[~np.isnan(leaders[:, COL_COMUNITY_REWARD]), COL_COMUNITY_REWARD].mean()

    # ゲーム実行
    for i in range(MAX_STEP):
        # コミュニティの模倣(移動)
        if i % COMUNITY_MOVE_TERM == 0:
            # comu_n_histry[int(i/COMUNITY_MOVE_TERM), :] = np.bincount(members[:, COL_COMUNITY].astype(np.int64), minlength=NUM_PLAYERS)
            # comu_r_histry[int(i/COMUNITY_MOVE_TERM), [0, 1, 2]] = leaders[:, COL_COMUNITY_REWARD] / COMUNITY_MOVE_TERM
            members[:, COL_COMUNITY] = get_newcomunity(leaders[:, COL_COMUNITY_REWARD], leaders[:, COL_PLAYERID], members.shape[0], mu=parameter['epsilon'])
            leaders[:, COL_COMUNITY_REWARD] = 0

        # 行動決定
        if i % LEADER_SAMPLING_TERM == 0:
            leaders[:, [COL_APC, COL_APS]], leaders[:, COL_ANUM]  = get_action_inpgg(leaders[:, [COL_Qap00, COL_Qap01, COL_Qap10, COL_Qap11]], epsilon=parameter['epsilon'])
            leaders[0, [COL_APC, COL_APS]] = 0
            leaders[0, COL_ANUM] = 0
        members[:, [COL_AC, COL_AS]], members[:, COL_ANUM] = get_action_inpgg(members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], epsilon=parameter['epsilon'])
        # 利得算出
        mrs = get_members_gain(
            members[:, COL_AC],
            members[:, COL_AS],
            members[:, COL_COMUNITY],
            leaders[:, COL_PLAYERID],
            leaders[:, COL_APC],
            leaders[:, COL_APS], 
            parameter
        )
        lrs = get_leaders_gain(
            members[:, COL_AC],
            members[:, COL_AS],
            members[:, COL_COMUNITY],
            leaders[:, COL_PLAYERID],
            leaders[:, COL_APC],
            leaders[:, COL_APS],
            parameter
        )

        members[:, COL_GAME_REWARD] += mrs
        leaders[:, COL_GAME_REWARD] += lrs
        members[:, COL_ROLE_REWARD] += mrs
        leaders[:, COL_ROLE_REWARD] += lrs

        # コミュニティーの評価の算出
        for cid in leaders[:, COL_PLAYERID]:
            cr = members[members[:, COL_COMUNITY] == cid, COL_GAME_REWARD]
            if cr.shape[0] != 0:
                leaders[leaders[:, COL_PLAYERID] == cid, COL_COMUNITY_REWARD] += np.mean(cr)
            else:
                leaders[leaders[:, COL_PLAYERID] == cid, COL_COMUNITY_REWARD] += 0.1

        # データ記録
        # qa_histry[i, :] = members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]].mean(axis=0)
        # qap_histry[i, :] = leaders[:, [COL_Qap00, COL_Qap01, COL_Qap10, COL_Qap11]].mean(axis=0)
        # 学習
        members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]] = learning_action(
            members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]],
            members[:, COL_GAME_REWARD],
            members[:, COL_ANUM],
            alpha=parameter['alpha']
        )
        members[:, COL_GAME_REWARD] = 0

        if i % LEADER_SAMPLING_TERM == LEADER_SAMPLING_TERM - 1:
            leaders[:, [COL_GAME_REWARD]] /= LEADER_SAMPLING_TERM
            leaders[:, [COL_Qap00, COL_Qap01, COL_Qap10, COL_Qap11]] = learning_action(
                leaders[:, [COL_Qap00, COL_Qap01, COL_Qap10, COL_Qap11]],
                leaders[:, COL_GAME_REWARD],
                leaders[:, COL_ANUM],
                alpha=parameter['alpha']
            )
            leaders[:, COL_GAME_REWARD] = 0

    players[players[:, COL_ROLE] == ROLE_MEMBER, :] = members
    players[players[:, COL_ROLE] == ROLE_LEADER, :] = leaders

    # return players, qa_histry, qap_histry, comu_n_histry, comu_r_histry
    return players

def get_players_role(players_qr, epsilon=0.02):
    '''
    abstract:
        epsilon-greedy法により次のゲームでのプレイヤーの役割を決定する
    input:
        players_qr:    np.array shape=[NUM_PLAYERS, 2(COL_QrLEADER, COL_QrMEMBERS)]
            全てのグループの成員固体
        epsilon:       float
            探索的行動をとる確率. default=0.02
    output:
        players_rule:   np.array shape=[NUM_PLAYERS]
            各プレイヤーが選択したゲームルール配列
    '''

    players_role = players_qr.argmax(axis=1)
    rand = np.random.rand(players_qr.shape[0])
    players_role[rand < epsilon] = np.random.choice([ROLE_LEADER, ROLE_MEMBER], size=(rand < epsilon).sum())

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
    mask[players_role == ROLE_LEADER, ROLE_LEADER] = 1
    mask[players_role == ROLE_MEMBER, ROLE_MEMBER] = 1

    # 更新
    error = mask * (np.tile(rewards,(2,1)).T - players_qr)

    return players_qr + ( alpha * error )