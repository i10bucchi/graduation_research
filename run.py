#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
import os
import sys
from model_helper import generate_players, one_order_game
from config import *
from make_batch_file import paramfilename
from multiprocessing import Pool

def process(seed, parameter, path):
    np.random.seed(seed=seed)

    thetas = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    for theta in thetas:
        players = generate_players()
        members = players[players['role'] == 'member'].values
        leader = players[players['role'] == 'leader'].values[0]

        # theta[0]: 成員の利益を考慮するか否か
        # theta[1]: 行動試行期間の差
        dfm, dfl, dftmp = one_order_game(members, leader, parameter, theta)
        
        pd.concat(dfm).to_csv(path + 'csv/{theta}_members_q_seed={seed}.csv'.format(theta=''.join([str(t) for t in theta]), seed=seed))
        pd.concat(dfl).to_csv(path + 'csv/{theta}_leader_q_seed={seed}.csv'.format(theta=''.join([str(t) for t in theta]), seed=seed))
        pd.concat(dftmp).to_csv(path + 'csv/{theta}_df_tmp_seed={seed}.csv'.format(theta=''.join([str(t) for t in theta]), seed=seed))

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

        path = rootpath + dirname + '/'
        arg = [(i, parameter, path) for i in range(S, MAX_REP)]
        with Pool(MULTI) as p:
            p.map_async(wrapper, arg).get(9999999)

if __name__== "__main__":
    main()
