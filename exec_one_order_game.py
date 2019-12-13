import model_helper
import config

players00 = model_helper.generate_players()
players01 = model_helper.generate_players()
players10 = model_helper.generate_players()
players11 = model_helper.generate_players()

parameter = {
    'cost_cooperate':   4,
    'cost_support':     2,
    'cost_punish':      2,
    'power_social':     4,
    'punish_size':      8,
    'alpha':            0.8,
    'epsilon':          0.05,
}

players00, a_rate00 = model_helper.one_order_game(players00, parameter, [0, 0])
players01, a_rate01 = model_helper.one_order_game(players01, parameter, [0, 1])
players10, a_rate10 = model_helper.one_order_game(players10, parameter, [1, 0])
players11, a_rate11 = model_helper.one_order_game(players11, parameter, [1, 1])

print('\n\n')
print('--------------------------------------')
print('             (c_rate, s_rate)')
print('a_rate 00: ', a_rate00)
print('a_rate 01: ', a_rate01)
print('a_rate 10: ', a_rate10)
print('a_rate 11: ', a_rate11)
print('--------------------------------------')
print('             rule rewards')
print('rule 00: ', players00[:, config.COL_ROLE_REWARD])
print('rule 01: ', players01[:, config.COL_ROLE_REWARD])
print('rule 10: ', players10[:, config.COL_ROLE_REWARD])
print('rule 11: ', players11[:, config.COL_ROLE_REWARD])
print('\n\n')