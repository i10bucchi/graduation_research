#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
import sys
import copy
from model_helper import generate_groups, get_next_leaders_number, divided_member_and_leaders_by_leaders_number, set_gene_s_by_pc_count, do_action, update_pc_count, calc_gain, evolution_members, evolution_leaders
from config import *
from make_batch_file import paramfilename
from multiprocessing import Pool

def process(seed, parameter, path):
    np.random.seed(seed=seed)

    groups = generate_groups()
    members_gene_ave = []
    leaders_gene_ave = []

    dfm = []
    dfl = []

    # 最初の制裁者決定
    step = MAX_SIMU
    leaders_number, pc_count = get_next_leaders_number(groups, parameter)
    member, leaders, is_groups, is_leaders = divided_member_and_leaders_by_leaders_number(groups, leaders_number)

    for i in tqdm(range(MAX_GENERATION)):
        # 制裁者決定
        if i % MAX_TERM_OF_OFFICE == MAX_TERM_OF_OFFICE - 1:
            # 結果をコミット
            groups[is_leaders] = leaders
            groups[is_groups] =  np.reshape(member, (member.shape[0]*member.shape[1], member.shape[2]))

            # 制裁者決定
            step = MAX_SIMU
            leaders_number, pc_count = get_next_leaders_number(groups, parameter)
            member, leaders, is_groups, is_leaders = divided_member_and_leaders_by_leaders_number(groups, leaders_number)
        
        # ゲーム
        for _ in range(MAX_GAME):
            member = set_gene_s_by_pc_count(member, pc_count, step)
            member, leaders = do_action(member, leaders)
            member, leaders = calc_gain(member, leaders, parameter)
            pc_count = update_pc_count(leaders, pc_count)
            step += 1
        
        # プロット用にログ記録
        members_gene_ave.append([member[:, :, COL_GC].mean(), member[:, :, COL_GS].mean()])
        leaders_gene_ave.append([leaders[:, COL_GPC].mean(), leaders[:, COL_GPS].mean()])

        df_c = pd.DataFrame(member[:, :, COL_GC])
        df_c['gene_name'] = 'c'
        df_c['generation'] = i
        df_c_copy = copy.deepcopy(df_c)
        df_s = pd.DataFrame(member[:, :, COL_GS])
        df_s['gene_name'] = 's'
        df_s['generation'] = i
        df_s_copy = copy.deepcopy(df_s)
        dfm.extend([df_c_copy, df_s_copy])

        df_pc = pd.DataFrame(leaders[:, COL_GPC])
        df_pc['gene_name'] = 'pc'
        df_pc['generation'] = i
        df_pc_copy = copy.deepcopy(df_pc)
        df_ps = pd.DataFrame(leaders[:, COL_GPS])
        df_ps['gene_name'] = 'ps'
        df_ps['generation'] = i
        df_ps_copy = copy.deepcopy(df_ps)
        dfl.extend([df_pc_copy, df_ps_copy])

        # 進化
        member = evolution_members(member)
        if i % FREQ_EVOL_LEADERS == FREQ_EVOL_LEADERS - 1:
            leaders = evolution_leaders(member, leaders)
        
        if i % (MAX_GENERATION / 10) == (MAX_GENERATION / 10) - 1:
            pd.concat(dfm).to_csv(path + 'csv/member_gene_seed={seed}_generation={i}.csv'.format(seed=seed, i=i))
            pd.concat(dfl).to_csv(path + 'csv/leader_gene_seed={seed}_generation={i}.csv'.format(seed=seed, i=i))
            dfm = []
            dfl = []
    
    # 結果保存
    pd.DataFrame(np.array(members_gene_ave)).to_csv(path + 'csv/groups_gene_ave_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(np.array(leaders_gene_ave)).to_csv(path + 'csv/leaders_gene_ave_seed={seed}.csv'.format(seed=seed))

# 引数を複数取るために必要
# https://qiita.com/kojpk/items/2919362de582a7d8de9e
def wrapper(arg):
    process(*arg)

def main():
    args = sys.argv
    rootpath = args[1]
    # parameterファイルを全て取得
    parameter_file_list = sorted(glob.glob("./parameter/*.yml"))

    # parameterファイルのパラメータ毎に実行
    for p_path in parameter_file_list:
        parameter = load_parameter(p_path)
        dirname = paramfilename(parameter)
        os.mkdir(rootpath + dirname)
        os.mkdir(rootpath + dirname + '/csv')

        p = Pool(MULTI)
        path = rootpath + dirname + '/'
        arg = [(i, parameter, path) for i in range(S, MAX_REP)]
        process(1, parameter, path)
        # p.map(wrapper, arg)
        # p.map_async(wrapper, arg).get(9999999)
        p.close

if __name__== "__main__":
    main()
