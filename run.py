#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
import os
import sys
import copy
from tqdm import tqdm
from model_helper import generate_players, exec_pgg, get_players_role, get_gaming_rule, get_rule_gain, learning_role
from config import *
from make_batch_file import paramfilename
from multiprocessing import Pool

def process(seed, parameter, path):
    np.random.seed(seed=seed)

    # エージェントの生成
    players = generate_players()

    # ゲームの実行
    for _ in tqdm(range(500)):
        # 制裁者としてゲームに参加するか成員としてゲームに参加するかの決定
        players[:, COL_ROLE] = get_players_role(players[:, [COL_QrLEADER, COL_QrMEMBERS]])

        # 共同罰あり公共財ゲームの実行
        players = exec_pgg(players, parameter)

        # 制裁者と成員の評価値算出
        players[:, [COL_QrLEADER, COL_QrMEMBERS]] = learning_role(
            players[:, [COL_QrLEADER, COL_QrMEMBERS]],
            players[:, COL_ROLE_REWARD]
        )
        
        # プロット用にログ記録
        # players_qr = players[:, [COL_QrLEADER, COL_Qr01, COL_Qr10, COL_QrMEMBERS]]
        # q_m_l.append(np.mean(players_qr[players[:, COL_ROLE] == ROLE_MEMBER, :], axis=0))
        # q_l_l.append(players_qr[players[:, COL_ROLE] == ROLE_LEADER, :][0])
        # m_r_l.append([players[players[:, COL_ROLE] == ROLE_MEMBER, COL_ROLE_REWARD].mean(), agreed_rule_number])
        # l_r_l.append([players[players[:, COL_ROLE] == ROLE_LEADER, COL_ROLE_REWARD][0], agreed_rule_number])
        # action_rate_l.append(action_rate)
    
    # 結果の保存
    # pd.DataFrame(q_m_l, columns=['Qr_00', 'Qr_01', 'Qr_10', 'Qr_11']).to_csv(path + 'csv/players_qrm_seed={seed}.csv'.format(seed=seed))
    # pd.DataFrame(q_l_l, columns=['Qr_00', 'Qr_01', 'Qr_10', 'Qr_11']).to_csv(path + 'csv/players_qrl_seed={seed}.csv'.format(seed=seed))
    # pd.DataFrame(m_r_l, columns=['reward', 'rule_number']).to_csv(path + 'csv/member_reward_seed={seed}.csv'.format(seed=seed))
    # pd.DataFrame(l_r_l, columns=['reward', 'rule_number']).to_csv(path + 'csv/leader_reward_seed={seed}.csv'.format(seed=seed))
    # pd.DataFrame(action_rate_l, columns=['cooperation_rate', 'supporting_rate']).to_csv(path + 'csv/players_action_rate_seed={seed}.csv'.format(seed=seed))

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
