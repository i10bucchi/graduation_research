import numpy as np
import pandas as pd
from model_helper import *
from config import *
from multiprocessing import Pool

np.set_printoptions(formatter={'float': '{: 0.1f}'.format})

def process(seed, leaders_points):
    np.random.seed(seed=seed)

    players = generate_players()
    players[:, COL_ROLE] = get_players_role(players[:, [COL_QrLEADER, COL_QrMEMBERS]])
    parameter = {
        'cost_cooperate':   4,
        'cost_support':     2,
        'cost_punish':      2,
        'power_social':     4,
        'punish_size':      8,
        'alpha':            0.8,
        'epsilon':          0.05,
    }


    # 制裁者としてゲームに参加するか成員としてゲームに参加するかの決定
    players[:, COL_ROLE] = ROLE_MEMBER
    players[leaders_points, COL_COMUNITY_REWARD] = 1

    players[leaders_points, COL_ROLE] = ROLE_LEADER

    # 共同罰あり公共財ゲームの実行
    players, qa_hist, qap_hist, cn_hist, cr_hist = exec_pgg(players, parameter)

    pd.DataFrame(qa_hist, columns=['Qa_00', 'Qa_01', 'Qa_10', 'Qa_11']).to_csv('result_pgg/members_qa_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(qap_hist, columns=['Qap_00', 'Qap_01', 'Qap_10', 'Qap_11']).to_csv('result_pgg/leaders_qap_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(cn_hist, columns=range(len(leaders_points))).to_csv('result_pgg/comunity_size_seed={seed}.csv'.format(seed=seed))
    pd.DataFrame(cr_hist, columns=range(len(leaders_points))).to_csv('result_pgg/comunity_reward_seed={seed}.csv'.format(seed=seed))

def wrapper(arg):
    process(*arg)

def main():
    arg = [(i, [0, 1, 2]) for i in range(S, MAX_REP)]
    with Pool(MULTI) as p:
        p.map_async(wrapper, arg).get(9999999)

if __name__== "__main__":
    main()