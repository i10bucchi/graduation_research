#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
import os
import sys
import copy
from tqdm import tqdm
from model_helper import generate_players, exec_pgg, get_players_role, learning_role
from config import *
from make_batch_file import paramfilename
from multiprocessing import Pool

def process(seed, parameter, path):
    np.random.seed(seed=seed)

    # エージェントの生成
    players = generate_players()

    qa_hist = np.zeros((MAX_TURN, 4))
    qap_hist = np.zeros((MAX_TURN, 4))
    cn_hist = np.zeros((MAX_TURN, NUM_PLAYERS))
    cr_hist = np.zeros((MAX_TURN, NUM_PLAYERS))
    role_hist = np.zeros((MAX_TURN, NUM_PLAYERS))
    # ゲームの実行
    for i in tqdm(range(MAX_TURN)):
        # 制裁者としてゲームに参加するか成員としてゲームに参加するかの決定
        if i == 0:
            players[:, COL_ROLE] = ROLE_MEMBER
            players[0, COL_COMUNITY_REWARD] = 1
        else:
            players[:, COL_ROLE] = get_players_role(players[:, [COL_QrLEADER, COL_QrMEMBERS]])
        players[0, COL_ROLE] = ROLE_LEADER

        # 共同罰あり公共財ゲームの実行
        players  = exec_pgg(players, parameter)

        # 制裁者と成員の評価値算出
        players[:, [COL_QrLEADER, COL_QrMEMBERS]] = learning_role(
            players[:, [COL_QrLEADER, COL_QrMEMBERS]],
            players[:, COL_ROLE_REWARD],
            players[:, COL_ROLE]
        )
        
        # プロット用にログ記録
        qa_hist[i, :] = players[:, [COL_Qa00, COL_Qa01, COL_Qa10, COL_Qa11]].mean(axis=0)
        qap_hist[i, :] = players[:, [COL_Qap00, COL_Qap01, COL_Qap10, COL_Qap11]].mean(axis=0)
        cn_hist[i, :] = np.bincount(players[:, COL_COMUNITY].astype(np.int64), minlength=NUM_PLAYERS)
        cr_hist[i, :] = players[:, COL_COMUNITY_REWARD] / COMUNITY_MOVE_TERM
        role_hist[i, :] = players[:, COL_ROLE]
    
    # 結果の保存
    pd.DataFrame(qa_hist, columns=['Qa_00', 'Qa_01', 'Qa_10', 'Qa_11']).to_csv(path + 'csv/players_qa_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(qap_hist, columns=['Qap_00', 'Qap_01', 'Qap_10', 'Qap_11']).to_csv(path + 'csv/players_qap_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(cn_hist, columns=range(NUM_PLAYERS)).to_csv(path + 'csv/comunity_population_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(cr_hist, columns=range(NUM_PLAYERS)).to_csv(path + 'csv/comunity_reward_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(role_hist, columns=range(NUM_PLAYERS)).to_csv(path + 'csv/role_seed={seed}.csv'.format(seed=seed))

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
