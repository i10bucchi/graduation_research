from model_helper import *
from config import *

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

players = exec_pgg(players, parameter)
print(players[:, COL_ROLE_REWARD])
players[:, [COL_QrLEADER, COL_QrMEMBERS]] = learning_role(
    players[:, [COL_QrLEADER, COL_QrMEMBERS]],
    players[:, COL_ROLE_REWARD],
    players[:, COL_ROLE]
)
print(players[:, [COL_QrLEADER, COL_QrMEMBERS]])