#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
import sys
import copy
from model_helper import generate_players, get_members_action, get_leader_action, calc_gain, learning_members, learning_leader
from config import *
from make_batch_file import paramfilename
from multiprocessing import Pool
import pandas as pd

def process(seed, parameter, path):
    np.random.seed(seed=seed)

    players = generate_players()
    members = players[players['role'] == 'member'].values
    leader = players[players['role'] == 'leader'].values[0]

    dfm = []
    dfl = []

    step = 0
    for i in tqdm(range(MAX_STEP)):
        # ゲーム
        if i % LEADER_SAMPLING_TERM == 0:
            leader = get_leader_action(leader)
        members = get_members_action(members)
        members, leader = calc_gain(members, leader, parameter)
        step += 1
        
        # プロット用にログ記録
        df = pd.DataFrame(members[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]], columns=['Qa_00', 'Qa_01', 'Qa_10', 'Qa_11'])
        df['step'] = i
        df['member_id'] = range(NUM_MEMBERS)
        df_copy = copy.deepcopy(df)
        dfm.append(df_copy)

        df = pd.DataFrame(np.array([leader[[COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]]]), columns=['Qa_00', 'Qa_01', 'Qa_10', 'Qa_11'])
        df['step'] = i
        df_copy = copy.deepcopy(df)
        dfl.append(df_copy)

        # 学習
        members = learning_members(members)
        members[COL_P_LOG] += members[COL_P]
        members[:, COL_P] = 0
        if i % LEADER_SAMPLING_TERM == LEADER_SAMPLING_TERM - 1:
            leader = learning_leader(members, leader, parameter)
            leader[COL_P] = 0
            members[:, COL_P_LOG] = 0
        
    pd.concat(dfm).to_csv(path + 'csv/members_q_seed={seed}.csv'.format(seed=seed))
    pd.concat(dfl).to_csv(path + 'csv/leader_q_seed={seed}.csv'.format(seed=seed))

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
        # process(1, parameter, path)
        # p.map(wrapper, arg)
        p.map_async(wrapper, arg).get(9999999)
        p.close

if __name__== "__main__":
    main()
