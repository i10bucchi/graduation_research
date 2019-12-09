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

def process(seed, parameter, path):
    np.random.seed(seed=seed)

    # [成員の利益を考慮するか否か, 行動試行期間の差]
    rule_dict = {
        0: [0, 0],
        1: [0, 1],
        2: [1, 0],
        3: [1, 1]
    }

    q_m_l = []
    q_l_l = []
    m_r_l = []
    l_r_l = []

    players = generate_players()
    for _ in tqdm(range(300)):
        agreed_rule_number = -1
        while agreed_rule_number == -1:
            players[:, COL_RNUM] = get_players_rule(players)
            agreed_rule_number = get_gaming_rule(players)
        theta = rule_dict[agreed_rule_number]

        players, _, _ = one_order_game(players, parameter, theta)

        players[:, [COL_Qr00, COL_Qr01, COL_Qr10, COL_Qr11]] = learning_rule(
            players[:, [COL_Qr00, COL_Qr01, COL_Qr10, COL_Qr11]],
            players[:, COL_RREWARD],
            agreed_rule_number
        )
        
        # プロット用にログ記録
        players_qr = players[:, [COL_Qr00, COL_Qr01, COL_Qr10, COL_Qr11]]
        q_m_l.append(np.mean(players_qr[players[:, COL_ROLE] == ROLE_MEMBER, :], axis=0))
        q_l_l.append(players_qr[players[:, COL_ROLE] == ROLE_LEADER, :][0])
        m_r_l.append([players[players[:, COL_ROLE] == ROLE_MEMBER, COL_RREWARD].mean(), agreed_rule_number])
        l_r_l.append([players[players[:, COL_ROLE] == ROLE_LEADER, COL_RREWARD][0], agreed_rule_number])
    
    pd.DataFrame(q_m_l, columns=['Qr_00', 'Qr_01', 'Qr_10', 'Qr_11']).to_csv(path + 'csv/players_qrm_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(q_l_l, columns=['Qr_00', 'Qr_01', 'Qr_10', 'Qr_11']).to_csv(path + 'csv/players_qrl_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(m_r_l, columns=['reward', 'rule_number']).to_csv(path + 'csv/member_reward_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(l_r_l, columns=['reward', 'rule_number']).to_csv(path + 'csv/leader_reward_seed={seed}.csv'.format(seed=seed))

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
