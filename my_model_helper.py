#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
from my_model_config import *

warnings.filterwarnings('error')


# 関数名:   generate_groups()
# 概要:     初期化された成員が属しているグループとそれらのグループ毎の制裁者を返す
# 引数:     --
# 返り値1:  np.array(agents) np.array dytype=float shape=[NUM_GROUPS, NUM_MEMBERS, カラム数]
def generate_groups():
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
    agents = [agents.values for i in range(NUM_GROUPS)]

    return np.array(agents)

# 関数名:   get_next_leaders
# 概要:     投票により次の制裁者を決定する
# 引数1:    agents np.array dytype=float shape=[グループ数, エージェント数, カラム数]
# 返り値1:  leaders_number list type=int shape=[グループ数]
# 返り値2:  np.array(pc_count) np.array type=int shape=[グループ数]
def get_next_leaders_number(agents):
    candidates = np.random.randint(0, NUM_MEMBERS, (NUM_GROUPS, NUM_CANDIDATES))
    expected_value = np.zeros((NUM_GROUPS, NUM_MEMBERS, NUM_CANDIDATES))
    
    assert agents.shape == (NUM_GROUPS, NUM_MEMBERS, NUM_COLUMN), 'expect: {0} actual: {1}'.format( (NUM_GROUPS, NUM_MEMBERS, 5), agents.shape )

    pc_counts = np.zeros( (NUM_GROUPS, NUM_CANDIDATES) )

    for i in range(NUM_CANDIDATES):
        groups, leaders, is_groups, is_leaders = divided_member_and_leaders_by_leaders_number(agents, candidates[:, i])

        for _ in range(MAX_SIMU):
            groups, leaders = do_action(groups, leaders)
            groups, leaders = calc_gain(groups, leaders)
            pc_counts[:, i] += leaders[:, COL_APC]
        agents[is_leaders] = leaders
        agents[is_groups] = np.reshape(groups, (groups.shape[0]*groups.shape[1], groups.shape[2]))
        g = set_gene_s_by_pc_count(agents, pc_counts[:, i], MAX_SIMU)
        expected_value[:, :, i] = g[:, :, COL_GS]
        # expected_value[:, :, i] = agents[:, :, COL_P]
        agents[:, :, COL_P] = 0
    
    hyou = np.argmax(expected_value, axis=2)

    leaders_number = []
    pc_count_next_leader = []
    for i in range(NUM_GROUPS):
        count = np.bincount(hyou[i])
        max_hyou = [j for j, x in enumerate(count) if x == max(count)]
        r = np.random.randint(0, len(max_hyou), 1)
        leaders_number.append(candidates[i, max_hyou[r[0]]])
        pc_count_next_leader.append(pc_counts[i, max_hyou[r[0]]])
        # if i == 0:
        #     for j in range(NUM_CANDIDATES):
        #         if leaders_number[i] == candidates[i, j]:
        #             print 'gene_pc: {:.4f} gene_ps: {:.4f} *'.format(agents[i, candidates[i, j], COL_GPC], agents[i, candidates[i, j], COL_GPS])
        #         else:
        #             print 'gene_pc: {:.4f} gene_ps: {:.4f}'.format(agents[i, candidates[i, j], COL_GPC], agents[i, candidates[i, j], COL_GPS])
        #     print count
        #     print pc_counts[i]
        #     print leaders_number, pc_count[i]
        #     print '------------------------------------------'
    return leaders_number, np.array(pc_count_next_leader)

def divided_member_and_leaders_by_leaders_number(agents, leaders_number):
    is_leaders = get_leaders_mask(leaders_number)
    is_groups = get_groups_mask(leaders_number)
    leaders = agents[is_leaders]
    groups = agents[is_groups]
    groups = np.reshape(groups, (NUM_GROUPS, NUM_MEMBERS-1, NUM_COLUMN))

    return groups, leaders, is_groups, is_leaders

# 関数名:   get_leaders_mask
# 概要:     制裁者の部分のみTrueになったagentsの大きさのBoolian配列を生成
# 引数1:    leaders_number (np.array or list) dytype=int shape=[グループ数]
# 返り値1:  is_leader np.array type=bool shape=[NUM_GROUPS, NUM_MEMBERS]
def get_leaders_mask(leaders_number):
    is_leaders = np.zeros((NUM_GROUPS, NUM_MEMBERS), dtype=bool)
    for i in range(NUM_GROUPS):
        is_leaders[i, leaders_number[i]] = True
    return is_leaders

# 関数名:   get_groups_mask
# 概要:     成員の部分のみTrueになったagentsの大きさのBoolian配列を生成
# 引数1:    leaders_number (np.array or list) dytype=int shape=[グループ数]
# 返り値1:  is_groups np.array type=bool shape=[NUM_GROUPS, NUM_MEMBERS]
def get_groups_mask(leaders_number):
    is_groups = np.ones((NUM_GROUPS, NUM_MEMBERS), dtype=bool)
    for i in range(NUM_GROUPS):
        is_groups[i, leaders_number[i]] = False
    return is_groups

# 関数名:   set_gene_s_by_pc_count
# 概要:     制裁回数と自己の協力傾向遺伝子から支援傾向遺伝子を決定
# 引数1:    groups np.array dytype=float shape=[グループ数, エージェント数, カラム数]
# 引数2:    pc_count np.array dtype=int shape=[グループ数]
# 引数3:    step int
# 返り値1:  groups np.array dytype=float shape=[グループ数, エージェント数, カラム数]
def set_gene_s_by_pc_count(groups, pc_count, step):
    step += MAX_SIMU
    pc_trend = pc_count / step
    pc_trend = np.tile(pc_trend,(groups.shape[1],1))
    pc_trend = pc_trend.transpose()
    # pcからcが乖離しているほどsは低くなる
    groups[:, :, COL_GS] = 1 - np.absolute(pc_trend - groups[:, :, COL_GC])

    return groups

# # 関数名:   do_action
# # 概要:     遺伝子から行動を確率的に決定し各成員と制裁者の行動カラムを書き換える
# # 引数1:    groups np.array dytype=float shape=[グループ数, エージェント数, カラム数]
# # 引数2:    leaders np.array dtype=float shape=[グループ数, カラム数]
# # 引数3:    leaders_number np.array dtype=int shape=[グループ数]
# # 返り値1:  groups np.array dytype=float shape=[グループ数, エージェント数, カラム数]
# # 返り値2:  leaders np.array dtype=float shape=[グループ数, カラム数]
def do_action(groups, leaders):
    # 行動 = 遺伝子 - 乱数[0, 1] + 1
    groups[:, :, COL_AC] = groups[:, :, COL_GC] - np.random.rand(groups.shape[0], groups.shape[1]) + np.ones((groups.shape[0], groups.shape[1]))
    groups[:, :, COL_AS] = groups[:, :, COL_GS] - np.random.rand(groups.shape[0], groups.shape[1]) + np.ones((groups.shape[0], groups.shape[1]))
    # 行動が1以上に1.0, 1未満に0.0を割り振る
    groups[:, :, COL_AC] = np.where(groups[:, :, COL_AC] < 1.0, 0.0, 1.0)
    groups[:, :, COL_AS] = np.where(groups[:, :, COL_AS] < 1.0, 0.0, 1.0)
    
    # 行動 = 遺伝子 - 乱数[0, 1] + 1
    leaders[:, COL_APC] = leaders[:, COL_GPC] - np.random.rand(leaders.shape[0]) + np.ones(leaders.shape[0])
    leaders[:, COL_APS] = leaders[:, COL_GPS] - np.random.rand(leaders.shape[0]) + np.ones(leaders.shape[0])
    # 行動が1以上に1.0, 1未満に0.0を割り振る
    leaders[:, COL_APC] = np.where(leaders[:, COL_APC] < 1.0, 0.0, 1.0)
    leaders[:, COL_APS] = np.where(leaders[:, COL_APS] < 1.0, 0.0, 1.0)

    return groups, leaders

# # 関数名:   get_members_gain
# # 概要:     各成員と制裁者の行動から成員の利得を算出
# # 引数1:    groups np.array dytype=float shape=[グループ数, エージェント数, カラム数]
# # 引数2:    leaders np.array dtype=float shape=[グループ数, カラム数]
# # 返り値1:  d + cp + sp - pcp - psp dytype=float shape=[グループ数, 成員数, カラム数]
def get_members_gain(groups, leaders):
    # 論理積用のマスク
    pcp_mask = np.array([[leaders[i, COL_APC]] * groups.shape[1] for i in range(groups.shape[0])])
    psp_mask = np.array([[leaders[i, COL_APS]] * groups.shape[1] for i in range(groups.shape[0])])

    # 成員の得点計算
    d = (COST_COOPERATE * np.sum(groups[:, :, COL_AC], axis=1) * POWER_SOCIAL) / (groups.shape[1])
    d = np.tile(d,(groups.shape[1],1))
    d = d.transpose()
    # 非協力の場合はCOST_COOPERATEがもらえる
    cp = (COST_COOPERATE * (np.ones((groups.shape[0], groups.shape[1])) - groups[:, :, COL_AC]))
    # 非支援の場合はCOST_SUPPORTがもらえる
    sp = (COST_SUPPORT * (np.ones((groups.shape[0], groups.shape[1])) - groups[:, :, COL_AS]))
    # 非協力の場合に制裁者が罰を行使してたら罰される
    pcp = (PUNISH_SIZE * np.logical_and(pcp_mask,  (np.ones((groups.shape[0], groups.shape[1])) - groups[:, :, COL_AC])))
    # 非支援の場合に制裁者が罰を行使してたら罰される
    psp = (PUNISH_SIZE * np.logical_and(psp_mask,  (np.ones((groups.shape[0], groups.shape[1])) - groups[:, :, COL_AS])))

    return d + cp + sp - pcp - psp

# 関数名:   get_leaders_gain
# 概要:     各成員と制裁者の行動から制裁者の利得を算出
# 引数1:    groups np.array dytype=float shape=[グループ数, エージェント数, カラム数]
# 引数2:    leaders np.array dtype=float shape=[グループ数, カラム数]
# 返り値1:  tax - pcc - psc dytype=float shape=[グループ数, カラム数]
def get_leaders_gain(groups, leaders):
    # 制裁者の得点計算
    tax = np.sum(groups[:, :, COL_AS], axis=1) * COST_SUPPORT
    assert tax.shape == (groups.shape[0],), 'expect: {0} actual: {1}'.format( tax.shape, (groups.shape[0]))
    # 非協力者制裁を行うコストを支払う
    pcc = COST_PUNISH * leaders[:, COL_APC] * (np.sum(np.ones((groups.shape[0], groups.shape[1])) - groups[:, :, COL_AC], axis=1))
    # 非支援者制裁を行うコストを支払う
    psc = COST_PUNISH * leaders[:, COL_APS] * (np.sum(np.ones((groups.shape[0], groups.shape[1])) - groups[:, :, COL_AS], axis=1))

    return tax - pcc - psc

# 関数名:   calc_gain
# 概要:     成員と制裁者の利得を算出しCOL_Pに加算して返す
# 引数1:    groups np.array dytype=float shape=[グループ数, エージェント数, カラム数]
# 引数2:    leaders np.array dtype=float shape=[グループ数, カラム数]
# 返り値1:  groups np.array dytype=float shape=[グループ数, エージェント数, カラム数]
def calc_gain(groups, leaders):
    groups[:, :, COL_P] += get_members_gain(groups, leaders)
    leaders[:, COL_P] += get_leaders_gain(groups, leaders)

    return groups, leaders

# 関数名:   softmax_2dim
# 概要:     与えられた配列をソフトマックスにかけて返す
# 引数1:    X np.array dtype=int, float shape=[len(X)]
# 返り値1:  expX / sum_expX np.array dtype=float shape=X.shape
def softmax_2dim(X):
    expX = np.exp(X)
    sum_expX = np.sum(expX, axis=1)
    sum_expX = np.tile(sum_expX, (X.shape[1],1))
    sum_expX = sum_expX.transpose()
    return expX / sum_expX


# 関数名:   softmax_1dim
# 概要:     与えられた配列をソフトマックスにかけて返す
# 引数1:    X np.array dtype=int, float shape=[len(X)]
# 返り値1:  expX / sum_expX np.array dtype=float shape=X.shape
def softmax_1dim(X):
    absmax = np.max(np.absolute(X))
    try:
        norm_X = X / absmax
    except RuntimeWarning as exe:
        norm_X = X
    expX = np.exp(norm_X)
    sum_expX = np.sum(expX)
    return expX / sum_expX

# # 関数名:   evolution_members
# # 概要:     成員の進化を行い新たな遺伝子へ遺伝子カラムを書き換えて返す
# # 引数1:    groups np.array dytype=float shape=[グループ数, 成員数, カラム数]
# # 返り値1:  groups np.array dytype=float shape=[グループ数, 成員数, カラム数]
def evolution_members(groups):
    # グループ内, グループ外どちらで進化を行うのか決める
    evol_mode = np.random.rand(groups.shape[0], groups.shape[1])
    evol_mode = np.where(evol_mode < PROB_EVOL_IN_GROUP, 1, 0)

    # ルーレットの目と盤を求める
    roulette_r = np.random.rand(groups.shape[0], groups.shape[1])
    norm_fitness = softmax_2dim(groups[:, :, COL_P])
    norm_fitness_cumsum = np.cumsum(norm_fitness, axis=1)

    norm_fitness_all = softmax_1dim(np.reshape(groups[:, :, COL_P], -1))
    norm_fitness_all_cumsum = np.cumsum(norm_fitness_all)

    next_groups = groups

    # 遺伝子の受け継ぎ
    for g, (r, nfc_g, evol_mode_g) in enumerate(zip(roulette_r, norm_fitness_cumsum, evol_mode)):
        for m, (r_m, evol_mode_m) in enumerate(zip (r, evol_mode_g)):
            if evol_mode_m == 1:
                index = np.where(nfc_g >= r_m)
                next_groups[g, m, (COL_GC, COL_GS)] = groups[g, index[0][0], (COL_GC, COL_GS)]
            else:
                index = np.where(norm_fitness_all_cumsum >= r_m)
                next_groups[g, m, (COL_GC, COL_GS)] = np.reshape(groups, (-1, NUM_COLUMN))[index[0][0], (COL_GC, COL_GS)]
    
    # 突然変異
    mutation_c = np.random.rand(groups.shape[0], groups.shape[1])
    mutation_s = np.random.rand(groups.shape[0], groups.shape[1])
    next_groups[:, :, COL_GC][mutation_c < PROB_MUTATION] = np.random.rand(next_groups[mutation_c < PROB_MUTATION].shape[0])
    next_groups[:, :, COL_GS][mutation_s < PROB_MUTATION] = np.random.rand(next_groups[mutation_s < PROB_MUTATION].shape[0])
    
    # ポイント初期化
    next_groups[:, :, COL_P_LOG] += next_groups[:, :, COL_P]
    next_groups[:, :, COL_P] = 0

    return next_groups

# # 関数名:   evolution_leaders
# # 概要:     制裁者の進化を行い新たな遺伝子へ遺伝子カラムを書き換えて返す
# # 引数1:    leaders np.array dytype=float shape=[グループ数, カラム数]
# # 返り値1:  leaders np.array dytype=float shape=[グループ数, カラム数]
def evolution_leaders(groups, leaders):
    # ルーレットの目と盤を求める
    roulette_r = np.random.rand(leaders.shape[0])
    fitness = softmax_1dim(leaders[:, COL_P]) * softmax_1dim(np.sum(groups[:, :, COL_P_LOG], axis=1))
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
    groups[:, :, COL_P_LOG] = 0
    next_leaders[:, COL_P] = 0

    return next_leaders

def save_result(plot_ave_g, plot_ave_l):
    df_gc = pd.DataFrame(data=plot_ave_g[:, (5000, 10000, 20000, 30000, 40000), 0], columns=['gc5000', 'gc10000', 'gc20000', 'gc30000', 'gc40000'], dtype='float')
    df_gs = pd.DataFrame(data=plot_ave_g[:, (5000, 10000, 20000, 30000, 40000), 1], columns=['gs5000', 'gs10000', 'gs20000', 'gs30000', 'gs40000'], dtype='float')
    df_lc = pd.DataFrame(data=plot_ave_l[:, (5000, 10000, 20000, 30000, 40000), 0], columns=['lc5000', 'lc10000', 'lc20000', 'lc30000', 'lc40000'], dtype='float')
    df_ls = pd.DataFrame(data=plot_ave_l[:, (5000, 10000, 20000, 30000, 40000), 1], columns=['ls5000', 'ls10000', 'ls20000', 'ls30000', 'ls40000'], dtype='float')
    df_all = pd.concat([df_gc, df_gs, df_lc, df_ls], axis=1)
    df_all.to_csv('./my_result_gene_scale_for_plot_hist.csv')

    df_gene_average_line_g = pd.DataFrame(data=plot_ave_g.mean(axis=0), columns=['gene_c', 'gene_s'], dtype='float')
    df_gene_average_line_l = pd.DataFrame(data=plot_ave_l.mean(axis=0), columns=['gene_pc', 'gene_ps'], dtype='float')
    df_all = pd.concat([df_gene_average_line_g, df_gene_average_line_l], axis=1)
    df_all.to_csv('./my_result_gene_average_for_plot_line.csv')