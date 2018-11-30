import yaml
import numpy as np

def paramfilename(parameter):
    filename = "{0}_{1}_{2}_{3}_{4}".format(
        parameter['cost_cooperate'],     # 0
        parameter['cost_support'],       # 1
        parameter['cost_punish'],        # 2
        parameter['power_social'],       # 3
        parameter['punish_size'],        # 4
    )

    return filename

set_cost_cooperate = [4]
set_cost_support = [2]
set_cost_punish = [2]
set_power_social = [4]
set_punish_size = [8]

for cost_cooperate in set_cost_cooperate:
    for cost_support in set_cost_support:
        for cost_punish in set_cost_punish:
            for power_social in set_power_social:
                for punish_size in set_punish_size:
                    parameter = {
                        'cost_cooperate':   cost_cooperate,
                        'cost_support':     cost_support,
                        'cost_punish':      cost_punish,
                        'power_social':     power_social,
                        'punish_size':      punish_size,
                    }

                    filename = paramfilename(parameter)
                    f = open('./parameter/' + filename + '.yml', 'w')
                    f.write(yaml.dump(parameter, default_flow_style=False))
                    f.close()