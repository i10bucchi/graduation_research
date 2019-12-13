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

# ゲームの実行
for i in range(MAX_TURN):
    # 制裁者としてゲームに参加するか成員としてゲームに参加するかの決定
    if i == 0:
        players[:, COL_ROLE] = ROLE_MEMBER
    else:
        players[:, COL_ROLE] = get_players_role(players[:, [COL_QrLEADER, COL_QrMEMBERS]])
    print('leaders_num: {}'.format(np.sum(players[:, COL_ROLE] == ROLE_LEADER)))
    print('members_num: {}'.format(np.sum(players[:, COL_ROLE] == ROLE_MEMBER)))

    # 共同罰あり公共財ゲームの実行
    players = exec_pgg(players, parameter)

    # 制裁者と成員の評価値算出
    players[:, [COL_QrLEADER, COL_QrMEMBERS]] = learning_role(
        players[:, [COL_QrLEADER, COL_QrMEMBERS]],
        players[:, COL_ROLE_REWARD],
        players[:, COL_ROLE]
    )