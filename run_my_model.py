#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
import os
import sys
from my_model_helper import generate_groups, get_next_leaders_number, divided_member_and_leaders_by_leaders_number, set_gene_s_by_pc_count, do_action, calc_gain, evolution_members, evolution_leaders
from my_model_config import *
from make_batch_file import paramfilename
from multiprocessing import Pool

def process(seed, parameter, path):
    np.random.seed(seed=seed)

    agents = generate_groups()
    groups_gene_ave = []
    leaders_gene_ave = []
    step = 0

    # 最初の制裁者決定
    leaders_number, pc_count = get_next_leaders_number(agents, parameter)
    groups, leaders, is_groups, is_leaders = divided_member_and_leaders_by_leaders_number(agents, leaders_number)

    for i in range(MAX_GENERATION):
        # 制裁者決定
        if i % MAX_TERM_OF_OFFICE == MAX_TERM_OF_OFFICE - 1:
            # 結果をコミット
            agents[is_leaders] = leaders
            agents[is_groups] =  np.reshape(groups, (groups.shape[0]*groups.shape[1], groups.shape[2]))

            # 制裁者決定
            leaders_number, pc_count = get_next_leaders_number(agents, parameter)
            groups, leaders, is_groups, is_leaders = divided_member_and_leaders_by_leaders_number(agents, leaders_number)
        
        # ゲーム
        for _ in range(MAX_GAME):
            groups = set_gene_s_by_pc_count(groups, pc_count, step % MAX_TERM_OF_OFFICE)
            groups, leaders = do_action(groups, leaders)
            groups, leaders = calc_gain(groups, leaders, parameter)
            step += 1
        
        # プロット用にログ記録
        groups_gene_ave.append([groups[:, :, COL_GC].mean(), groups[:, :, COL_GS].mean()])
        leaders_gene_ave.append([leaders[:, COL_GPC].mean(), leaders[:, COL_GPS].mean()])
        pd.DataFrame(groups[:, :, COL_GC]).to_csv(path + 'csv/groups_gene_c_g={g}_seed={seed}.csv'.format(g=i, seed=seed))
        pd.DataFrame(groups[:, :, COL_GS]).to_csv(path + 'csv/groups_gene_s_g={g}_seed={seed}.csv'.format(g=i, seed=seed))
        pd.DataFrame(leaders[:, COL_GPC]).to_csv(path + 'csv/leaders_gene_pc_g={g}_seed={seed}.csv'.format(g=i, seed=seed))
        pd.DataFrame(leaders[:, COL_GPS]).to_csv(path + 'csv/leaders_gene_ps_g={g}_seed={seed}.csv'.format(g=i, seed=seed))

        # 進化
        groups = evolution_members(groups)
        if i % FREQ_EVOL_LEADERS == FREQ_EVOL_LEADERS - 1:
            leaders = evolution_leaders(groups, leaders)
    
    # 結果保存
    pd.DataFrame(np.array(groups_gene_ave)).to_csv(path + 'csv/groups_gene_ave_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(np.array(leaders_gene_ave)).to_csv(path + 'csv/leaders_gene_ave_seed={seed}.csv'.format(seed=seed))
    print('{}/{} -done'.format(seed, MAX_REP))

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

        p = Pool(4)
        path = rootpath + dirname + '/'
        arg = [(i, parameter, path) for i in range(MAX_REP)]
        p.map(wrapper, arg)
        p.close

if __name__== "__main__":
    main()