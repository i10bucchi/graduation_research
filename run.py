#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
import os
import sys
import copy
import pandas 
from tqdm import tqdm
from model_helper import generate_players, exec_game
from config import *
from make_batch_file import paramfilename
from multiprocessing import Pool

def process(seed, parameter, path):
    np.random.seed(seed=seed)

    players = generate_players()
    players, df = exec_game(players, parameter)

    df.to_csv(path + f'csv/result_action_num_seed={seed}.csv')

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
