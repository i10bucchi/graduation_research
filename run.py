#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
import os
import sys
import copy
from tqdm import tqdm
from model_helper import generate_players, one_order_game, get_players_rule, get_gaming_rule, get_rule_gain, learning_rule
from config import *
from make_batch_file import paramfilename
from multiprocessing import Pool
import pickle

def process(seed, parameter, path):
    np.random.seed(seed=seed)

    # [成員の利益を考慮するか否か, 行動試行期間の差]
    rule_dict = {
        0: [0, 0],
        1: [0, 1],
        2: [1, 0],
        3: [1, 1]
    }

    # qma_hist_list = []
    # qla_hist_list = []
    qr_hist_list = []
    q_m_l = []
    q_m_std_l = []
    q_m_median_l = []
    q_l_l = []
    rule_hist = []
    action_rate_l = []

    players = generate_players()
    for turn in tqdm(range(500)):
        agreed_rule_number = -1
        while agreed_rule_number == -1:
            players[:, COL_RNUM] = get_players_rule(players, epshilon=A+(B/(turn+1)))
            agreed_rule_number = get_gaming_rule(players)
        theta = rule_dict[agreed_rule_number]

        players, action_rate, qma_hist, qla_hist = one_order_game(players, parameter, theta)

        players[:, [COL_Qr00, COL_Qr01, COL_Qr10, COL_Qr11]] = learning_rule(
            players[:, [COL_Qr00, COL_Qr01, COL_Qr10, COL_Qr11]],
            players[:, COL_RREWARD],
            agreed_rule_number
        )
        
        # プロット用にログ記録
        # qma_hist_list.append(qma_hist)
        # qla_hist_list.append(qla_hist)
        players_qr = players[:, [COL_Qr00, COL_Qr01, COL_Qr10, COL_Qr11]]
        qr_hist_list.append(players_qr)
        q_m_l.append(np.mean(players_qr[players[:, COL_ROLE] == ROLE_MEMBER, :], axis=0))
        q_m_std_l.append(np.std(players_qr[players[:, COL_ROLE] == ROLE_MEMBER, :], axis=0))
        q_m_median_l.append(np.median(players_qr[players[:, COL_ROLE] == ROLE_MEMBER, :], axis=0))
        q_l_l.append(players_qr[players[:, COL_ROLE] == ROLE_LEADER, :][0])
        rule_hist.append(agreed_rule_number)
        action_rate_l.append(action_rate)
    
    # with open(path + f'qma_hist_seed={seed}.pickle', 'wb') as f:
    #     pickle.dump(qma_hist_list, f)

    # with open(path + f'qla_hist_seed={seed}.pickle', 'wb') as f:
    #     pickle.dump(qla_hist_list, f)

    with open(path + f'qr_hist_seed={seed}.pickle', 'wb') as f:
        pickle.dump(qr_hist_list, f)

    pd.DataFrame(q_m_l, columns=['Qr_00', 'Qr_01', 'Qr_10', 'Qr_11']).to_csv(path + 'csv/players_qrm_mean_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(q_m_std_l, columns=['Qr_00', 'Qr_01', 'Qr_10', 'Qr_11']).to_csv(path + 'csv/players_qrm_std_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(q_l_l, columns=['Qr_00', 'Qr_01', 'Qr_10', 'Qr_11']).to_csv(path + 'csv/players_qrl_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(rule_hist, columns=['rule']).to_csv(path + 'csv/rule_hist_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(action_rate_l, columns=['cooperation_rate', 'supporting_rate']).to_csv(path + 'csv/action_rate_seed={seed}.csv'.format(seed=seed))

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
        target_seed = [1, 3, 4, 2, 5, 6]
        # target_seed = range(80)
        arg = [(i, parameter, path) for i in target_seed]
        with Pool(MULTI) as p:
            p.map_async(wrapper, arg).get(9999999)

if __name__== "__main__":
    main()
