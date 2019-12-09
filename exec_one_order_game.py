import pandas as pd
import numpy as np
import model_helper
import config
from multiprocessing import Pool

parameter = {
    'cost_cooperate':   4,
    'cost_support':     2,
    'cost_punish':      2,
    'power_social':     4,
    'punish_size':      8,
    'alpha':            0.8,
    'epsilon':          0.05,
}
def process(seed, parameter):
    np.random.seed(seed=seed)
    rules = [
        ([0,0], '00'),
        ([0,1], '01'),
        ([1,0], '10'),
        ([1,1], '11')
    ]
    for rule, rule_str in rules:
        players = model_helper.generate_players()
        players, mqa_hist, lqa_hist = model_helper.one_order_game(players, parameter, rule)
        columns=['000', '001', '010', '011', '100', '101', '110', '111']
        df_mqa = pd.DataFrame(mqa_hist.mean(axis=1), columns=columns)
        df_mqa.to_csv(f'result_pgg/members_qvalue_hist_rule={rule_str}_seed={seed}.csv')
        columns=['00', '01', '10', '11']
        df_lqa = pd.DataFrame(lqa_hist, columns=columns)
        df_lqa.to_csv(f'result_pgg/leader_qvalue_hist_rule={rule_str}_seed={seed}.csv')

def wrapper(arg):
    process(*arg)

def main():
    arg = [(i, parameter) for i in range(config.S, config.MAX_REP)]
    with Pool(config.MULTI) as p:
        p.map_async(wrapper, arg).get(9999999)

if __name__== "__main__":
    main()