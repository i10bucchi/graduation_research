#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
from config import *

warnings.filterwarnings('error')


def generate_groups():
    '''
    abstract:
        各遺伝子を持った成員をNUM_MEMBER人含むグループをNUM_GROUP個作成
    input:
        --
    output:
        np.array(groups): np.array shape=[NUM_GROUPS, NUM_MEMBERS, NUM_COLUMN]
            全てのグループの個体
    '''

    groups = []
    for _ in range(NUM_GROUPS):
        agents = pd.DataFrame(columns=[])
        agents['point'] = [0 for i in range(NUM_MEMBERS)]
        agents['point_log'] = [0 for i in range(NUM_MEMBERS)]
        agents['action_c'] = [-1 for i in range(NUM_MEMBERS)]
        agents['action_s'] = [-1 for i in range(NUM_MEMBERS)]
        agents['action_pc'] = [-1 for i in range(NUM_MEMBERS)]
        agents['action_ps'] = [-1 for i in range(NUM_MEMBERS)]
        agents['gene_c'] = np.random.rand(NUM_MEMBERS)
        agents['gene_s'] = np.random.rand(NUM_MEMBERS)
        agents['gene_pc'] = np.random.rand(NUM_MEMBERS)
        agents['gene_ps'] = np.random.rand(NUM_MEMBERS)
        groups.append(agents.values)

    return np.array(groups)

def get_next_leaders_number(groups, parameter):
    '''
    abstract:
        投票によって次の制裁者を決定する
    input:
        groups:     np.array shape=[NUM_GROUPS, NUM_MEMBERS, NUM_COLUMN]
            全てのグループの個体
        parameter:  dict 
            実験パラメータ
    output:
        leader_number:                  list shape=[NUM_GROUPS]
            各グループの次の制裁者の個体番号(arrayのindex)
        np.array(pc_count_next_leader): type=np.array shape=[NUM_GROUPS]
            各グループの次の制裁者がお試し期間で行なった制裁の数
    '''

    # 立候補者をランダムで取得
    candidates = np.random.randint(0, NUM_MEMBERS, (NUM_GROUPS, NUM_CANDIDATES))
    # 各立候補者に対する期待値を初期化
    expected_value = np.zeros((NUM_GROUPS, NUM_MEMBERS, NUM_CANDIDATES))

    pc_counts = np.zeros( (NUM_GROUPS, NUM_CANDIDATES) )

    # 各立候補者を制裁者に据えてお試し期間を行う
    for i in range(NUM_CANDIDATES):
        members, leaders, is_members, is_leaders = divided_member_and_leaders_by_leaders_number(groups, candidates[:, i])

        for _ in range(MAX_SIMU):
            members, leaders = do_action(members, leaders)
            members, leaders = calc_gain(members, leaders, parameter)
            pc_counts[:, i] += leaders[:, COL_APC]
        groups[is_leaders] = leaders
        groups[is_members] = np.reshape(members, (members.shape[0]*members.shape[1], members.shape[2]))
        g = set_gene_s_by_pc_count(groups, pc_counts[:, i], MAX_SIMU)
        expected_value[:, :, i] = g[:, :, COL_GS]
        # expected_value[:, :, i] = members[:, :, COL_P]
        groups[:, :, COL_P] = 0
    
    hyou = np.argmax(expected_value, axis=2)

    leaders_number = []
    pc_count_next_leader = []
    for i in range(NUM_GROUPS):
        count = np.bincount(hyou[i])
        max_hyou = [j for j, x in enumerate(count) if x == max(count)]
        r = np.random.randint(0, len(max_hyou), 1)
        leaders_number.append(candidates[i, max_hyou[r[0]]])
        pc_count_next_leader.append(pc_counts[i, max_hyou[r[0]]])

    return leaders_number, np.array(pc_count_next_leader)

def divided_member_and_leaders_by_leaders_number(groups, leaders_number):
    '''
    abstract:
        グループから成員と制裁者を分ける
    input:
        groups:         np.array shape=[NUM_GROUPS, NUM_MEMBERS, NUM_COLUMN]
            全てのグループの個体
        leader_number:  list shape=[NUM_GROUPS]
            各グループの次の制裁者の個体番号(arrayのindex)
    output:
        members:    np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
            全てのグループの成員固体
        leaders:    np.array shape=[NUM_GROUPS, NUM_COLUMN]
            全てのグループの制裁者固体
        is_members: np.array shape=[NUM_GROUPS, NUM_MEMBERS]
            全てのグループの成員固体の場所が1, 制裁者個体の場所が0のマスク
        is_leaders: np.array shape=[NUM_GROUPS, NUM_MEMBERS]
            全てのグループの制裁者個体の場所が1, 成員個体の場所が0のマスク
    '''

    is_leaders = get_leaders_mask(leaders_number)
    is_members = get_groups_mask(leaders_number)
    leaders = groups[is_leaders]
    members = groups[is_members]
    members = np.reshape(members, (NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN))

    return members, leaders, is_members, is_leaders

def get_leaders_mask(leaders_number):
    '''
    abstract:
        制裁者の場所のみTrueとなった制裁者抽出用マスクを生成する
    input:
        leader_number:  list shape=[NUM_GROUPS]
            各グループの次の制裁者の個体番号(arrayのindex)
    output:
        is_leaders: np.array shape=[NUM_GROUPS, NUM_MEMBERS]
            全てのグループの制裁者個体の場所が1, 成員個体の場所が0のマスク
    '''

    is_leaders = np.zeros((NUM_GROUPS, NUM_MEMBERS), dtype=bool)
    for i in range(NUM_GROUPS):
        is_leaders[i, leaders_number[i]] = True
    return is_leaders

def get_groups_mask(leaders_number):
    '''
    abstract:
        成員の場所のみTrueとなった成員抽出用マスクを生成する
    input:
        leader_number:  list shape=[NUM_GROUPS]
            各グループの次の制裁者の個体番号(arrayのindex)
    output:
        is_members: np.array shape=[NUM_GROUPS, NUM_MEMBERS]
            全てのグループの成員個体の場所が1, 制裁者個体の場所が0のマスク
    '''

    is_members = np.ones((NUM_GROUPS, NUM_MEMBERS), dtype=bool)
    for i in range(NUM_GROUPS):
        is_members[i, leaders_number[i]] = False
    return is_members


def set_gene_s_by_pc_count(groups, pc_count, step):
    '''
    abstract:
        制裁回数と自己の協力傾向遺伝子から支援傾向遺伝子直を決定しセットする
    input:
        groups:         np.array shape=[NUM_GROUPS, NUM_MEMBERS, NUM_COLUMN]
            全てのグループの個体
        pc_count:   np.array length=NUM_GROUPS
            現在の制裁者が就任してから制裁を行なった回数(お試し期間も含む)
        step:       int
            現在の制裁者が就任してから行なったゲーム数(お試し期間も含む)
    output:
        groups:         np.array shape=[NUM_GROUPS, NUM_MEMBERS, NUM_COLUMN]
            全てのグループの個体
    '''

    pc_trend = pc_count / step
    pc_trend = np.tile(pc_trend,(groups.shape[1],1))
    pc_trend = pc_trend.transpose()
    # pcからcが乖離しているほどsは低くなる
    groups[:, :, COL_GS] = 1 - np.absolute(pc_trend - groups[:, :, COL_GC])

    return groups

def do_action(members, leaders):
    '''
    abstract:
        遺伝子から行動を確率的に決定して各成員と制裁者の行動カラムを書き換える
    input:
        members:    np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
            全てのグループの成員固体
        leaders:    np.array shape=[NUM_GROUPS, NUM_COLUMN]
            全てのグループの制裁者固体
    output:
        members:    np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
            全てのグループの成員固体
        leaders:    np.array shape=[NUM_GROUPS, NUM_COLUMN]
            全てのグループの制裁者固体
    '''

    # 行動 = 遺伝子 - 乱数[0, 1] + 1
    members[:, :, COL_AC] = members[:, :, COL_GC] - np.random.rand(members.shape[0], members.shape[1]) + np.ones((members.shape[0], members.shape[1]))
    members[:, :, COL_AS] = members[:, :, COL_GS] - np.random.rand(members.shape[0], members.shape[1]) + np.ones((members.shape[0], members.shape[1]))
    # 行動が1以上に1.0, 1未満に0.0を割り振る
    members[:, :, COL_AC] = np.where(members[:, :, COL_AC] < 1.0, 0.0, 1.0)
    members[:, :, COL_AS] = np.where(members[:, :, COL_AS] < 1.0, 0.0, 1.0)
    
    # 行動 = 遺伝子 - 乱数[0, 1] + 1
    leaders[:, COL_APC] = leaders[:, COL_GPC] - np.random.rand(leaders.shape[0]) + np.ones(leaders.shape[0])
    leaders[:, COL_APS] = leaders[:, COL_GPS] - np.random.rand(leaders.shape[0]) + np.ones(leaders.shape[0])
    # 行動が1以上に1.0, 1未満に0.0を割り振る
    leaders[:, COL_APC] = np.where(leaders[:, COL_APC] < 1.0, 0.0, 1.0)
    leaders[:, COL_APS] = np.where(leaders[:, COL_APS] < 1.0, 0.0, 1.0)

    return members, leaders

def update_pc_count(leaders, pc_count):
    '''
    abstract:
        制裁回数を更新
    input:
        leaders:    np.array shape=[NUM_GROUPS, NUM_COLUMN]
            全てのグループの制裁者固体
        pc_count:   np.array length=NUM_GROUPS
            現在の制裁者が就任してから制裁を行なった回数(お試し期間も含む)
    output:
        pc_count:   np.array length=NUM_GROUPS
            現在の制裁者が就任してから制裁を行なった回数(お試し期間も含む)
    '''

    pc_count = pc_count + leaders[:, COL_APC]
    return pc_count

def get_members_gain(members, leaders, parameter):
    '''
    abstract:
        各成員と制裁者の行動から成員の利得を算出
    input:
        members:    np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
            全てのグループの成員固体
        leaders:    np.array shape=[NUM_GROUPS, NUM_COLUMN]
            全てのグループの制裁者固体
        parameter:  dict 
            実験パラメータ
    output:
        d + cp + sp - pcp - psp: np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
    '''

    # 論理積用のマスク
    pcp_mask = np.array([[leaders[i, COL_APC]] * members.shape[1] for i in range(members.shape[0])])
    psp_mask = np.array([[leaders[i, COL_APS]] * members.shape[1] for i in range(members.shape[0])])

    # 成員の得点計算
    d = (parameter['cost_cooperate'] * np.sum(members[:, :, COL_AC], axis=1) * parameter['power_social']) / (members.shape[1])
    d = np.tile(d,(members.shape[1],1))
    d = d.transpose()
    # 非協力の場合はparameter['cost_cooperate']がもらえる
    cp = (parameter['cost_cooperate'] * (np.ones((members.shape[0], members.shape[1])) - members[:, :, COL_AC]))
    # 非支援の場合はparameter['cost_support']がもらえる
    sp = (parameter['cost_support'] * (np.ones((members.shape[0], members.shape[1])) - members[:, :, COL_AS]))
    # 非協力の場合に制裁者が罰を行使してたら罰される
    pcp = (parameter['punish_size'] * np.logical_and(pcp_mask,  (np.ones((members.shape[0], members.shape[1])) - members[:, :, COL_AC])))
    # 非支援の場合に制裁者が罰を行使してたら罰される
    psp = (parameter['punish_size'] * np.logical_and(psp_mask,  (np.ones((members.shape[0], members.shape[1])) - members[:, :, COL_AS])))

    return d + cp + sp - pcp - psp

def get_leaders_gain(groups, leaders, parameter):
    '''
    abstract:
        各成員と制裁者の行動から制裁者の利得を算出
    input:
        members:    np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
            全てのグループの成員固体
        leaders:    np.array shape=[NUM_GROUPS, NUM_COLUMN]
            全てのグループの制裁者固体
        parameter:  dict 
            実験パラメータ
    output:
        tax - pcc - psc: np.array shape=[NUM_GROUPS, NUM_COLUMN]
    '''

    # 制裁者の得点計算
    tax = np.sum(groups[:, :, COL_AS], axis=1) * parameter['cost_support']
    # 非協力者制裁を行うコストを支払う
    pcc = parameter['cost_punish'] * leaders[:, COL_APC] * (np.sum(np.ones((groups.shape[0], groups.shape[1])) - groups[:, :, COL_AC], axis=1))
    # 非支援者制裁を行うコストを支払う
    psc = parameter['cost_punish'] * leaders[:, COL_APS] * (np.sum(np.ones((groups.shape[0], groups.shape[1])) - groups[:, :, COL_AS], axis=1))

    return tax - pcc - psc

def calc_gain(members, leaders, parameter):
    '''
    abstract:
        成員と制裁者の利得を算出しCOL_Pに加算して返す
    input:
        members:    np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
            全てのグループの成員固体
        leaders:    np.array shape=[NUM_GROUPS, NUM_COLUMN]
            全てのグループの制裁者固体
        parameter:  dict 
            実験パラメータ
    output:
        members:    np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
            全てのグループの成員固体
        leaders:    np.array shape=[NUM_GROUPS, NUM_COLUMN]
            全てのグループの制裁者固体
    '''

    members[:, :, COL_P] += get_members_gain(members, leaders, parameter)
    leaders[:, COL_P] += get_leaders_gain(members, leaders, parameter)

    return members, leaders

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

def evolution_members(members):
    '''
    abstract:
        成員の進化を行う
    input:
        members:    np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
            全てのグループの成員固体
    output:
        next_members:    np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
            進化した全てのグループの成員固体
    '''

    # グループ内, グループ外どちらを比較対象としてで進化を行うのか決める
    evol_mode = np.random.rand(members.shape[0], members.shape[1])
    evol_mode = np.where(evol_mode < PROB_EVOL_IN_GROUP, 1, 0)

    # ルーレットの目と盤を求める
    roulette_r = np.random.rand(members.shape[0], members.shape[1])
    norm_fitness = softmax_2dim(members[:, :, COL_P])
    norm_fitness_cumsum = np.cumsum(norm_fitness, axis=1)

    norm_fitness_all = softmax_1dim(np.reshape(members[:, :, COL_P], -1))
    norm_fitness_all_cumsum = np.cumsum(norm_fitness_all)

    next_members = members

    # 遺伝子の受け継ぎ
    for g, (r, nfc_g, evol_mode_g) in enumerate(zip(roulette_r, norm_fitness_cumsum, evol_mode)):
        for m, (r_m, evol_mode_m) in enumerate(zip (r, evol_mode_g)):
            if evol_mode_m == 1:
                index = np.where(nfc_g >= r_m)
                next_members[g, m, (COL_GC, COL_GS)] = members[g, index[0][0], (COL_GC, COL_GS)]
            else:
                index = np.where(norm_fitness_all_cumsum >= r_m)
                next_members[g, m, (COL_GC, COL_GS)] = np.reshape(members, (-1, NUM_COLUMN))[index[0][0], (COL_GC, COL_GS)]
    
    # 突然変異
    mutation_c = np.random.rand(members.shape[0], members.shape[1])
    mutation_s = np.random.rand(members.shape[0], members.shape[1])
    next_members[:, :, COL_GC][mutation_c < PROB_MUTATION] = np.random.rand(next_members[mutation_c < PROB_MUTATION].shape[0])
    next_members[:, :, COL_GS][mutation_s < PROB_MUTATION] = np.random.rand(next_members[mutation_s < PROB_MUTATION].shape[0])
    
    # ポイント初期化
    next_members[:, :, COL_P_LOG] += next_members[:, :, COL_P]
    next_members[:, :, COL_P] = 0

    return next_members

# # 関数名:   evolution_leaders
# # 概要:     制裁者の進化を行い新たな遺伝子へ遺伝子カラムを書き換えて返す
# # 引数1:    leaders np.array dytype=float shape=[グループ数, カラム数]
# # 返り値1:  leaders np.array dytype=float shape=[グループ数, カラム数]
def evolution_leaders(members, leaders):
    '''
    abstract:
        制裁者の進化を行う
    input:
        members:    np.array shape=[NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN]
            全てのグループの成員固体
        leaders:    np.array shape=[NUM_GROUPS, NUM_COLUMN]
            全てのグループの制裁者固体
    output:
        next_leaders: np.array shape=[NUM_GROUPS, NUM_COLUMN]
            進化した全てのグループの制裁者個体
    '''

    # ルーレットの目と盤を求める
    roulette_r = np.random.rand(leaders.shape[0])
    fitness = softmax_1dim(leaders[:, COL_P]) * softmax_1dim(np.sum(members[:, :, COL_P_LOG], axis=1))
    norm_fitness = softmax_1dim(fitness)
    norm_fitness_cumsum = np.cumsum(norm_fitness)

    # 遺伝子の受け継ぎ
    next_leaders = leaders
    for i, r in enumerate(roulette_r):
        index = np.where(norm_fitness_cumsum >= r)
        next_leaders[i, (COL_GPC, COL_GPS)] = leaders[index[0][0], (COL_GPC, COL_GPS)]
    
    # 突然変異
    mutation_pc = np.random.rand(leaders.shape[0])
    mutation_ps = np.random.rand(leaders.shape[0])
    next_leaders[:, COL_GPC][mutation_pc < PROB_MUTATION] = np.random.rand(next_leaders[mutation_pc < PROB_MUTATION].shape[0])
    next_leaders[:, COL_GPS][mutation_ps < PROB_MUTATION] = np.random.rand(next_leaders[mutation_ps < PROB_MUTATION].shape[0])

    # ポイントの初期化
    members[:, :, COL_P_LOG] = 0
    next_leaders[:, COL_P] = 0

    return next_leaders
