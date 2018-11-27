#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
from my_model_helper import generate_groups, get_next_leaders_number, divided_member_and_leaders_by_leaders_number, set_gene_s_by_pc_count, do_action, calc_gain, evolution_members, evolution_leaders, save_result
from my_model_config import *
from multiprocessing import Pool

def process(seed):
    np.random.seed(seed=seed)

    agents = generate_groups()
    groups_gene_ave = []
    leaders_gene_ave = []
    step = 0

    # 最初の制裁者決定
    leaders_number, pc_count = get_next_leaders_number(agents)
    groups, leaders, is_groups, is_leaders = divided_member_and_leaders_by_leaders_number(agents, leaders_number)

    for i in tqdm(range(MAX_GENERATION)):
        # 制裁者決定
        if i % MAX_TERM_OF_OFFICE == MAX_TERM_OF_OFFICE - 1:
            # 結果をコミット
            agents[is_leaders] = leaders
            agents[is_groups] =  np.reshape(groups, (groups.shape[0]*groups.shape[1], groups.shape[2]))

            # 制裁者決定
            leaders_number, pc_count = get_next_leaders_number(agents)
            groups, leaders, is_groups, is_leaders = divided_member_and_leaders_by_leaders_number(agents, leaders_number)
        
        # ゲーム
        for _ in range(MAX_GAME):
            groups = set_gene_s_by_pc_count(groups, pc_count, step % MAX_TERM_OF_OFFICE)
            groups, leaders = do_action(groups, leaders)
            groups, leaders = calc_gain(groups, leaders)
            step += 1
        
        # プロット用にログ記録
        groups_gene_ave.append([groups[:, :, COL_GC].mean(), groups[:, :, COL_GS].mean()])
        leaders_gene_ave.append([leaders[:, COL_GPC].mean(), leaders[:, COL_GPS].mean()])

        # 進化
        groups = evolution_members(groups)
        if i % FREQ_EVOL_LEADERS == FREQ_EVOL_LEADERS - 1:
            leaders = evolution_leaders(groups, leaders)
    
    pd.DataFrame(np.array(groups_gene_ave)).to_csv('groups_gene_ave_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(np.array(leaders_gene_ave)).to_csv('leaders_gene_ave_seed={seed}.csv'.format(seed=seed))

if __name__== "__main__":
    p = Pool(4)
    p.map( process, range(4) )
    # for i in range(4):
    #     process(i)