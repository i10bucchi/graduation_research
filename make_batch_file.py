#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import numpy as np
from config import *

def paramfilename(parameter):
    '''
    abstract:
        パラメータが格納されている辞書からファイル名を作成
    input:
        parameter dict
            実験パラメータ
    output:
        filename str
            ファイル名
    '''
    filename = "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(
        parameter['cost_cooperate'],     # 0
        parameter['cost_support'],       # 1
        parameter['cost_punish'],        # 2
        parameter['power_social'],       # 3
        parameter['punish_size'],        # 4
        parameter['alpha'],              # 5
        parameter['epsilon'],            # 6
    )

    return filename

def make_parameter_file():
    '''
    abstract:
        実験用のバッチファイルを生成する
    input:
        --
    output:
        --
    '''
    for cost_cooperate in COST_C_LIST:
        for cost_support in COST_S_LIST:
            for cost_punish in COST_P_LIST:
                for power_social in SP_LIST:
                    for punish_size in PUNISH_SIZE_LIST:
                        for alpha in ALPHA_LIST:
                            for epsilon in EPSILON_LIST:
                                parameter = {
                                    'cost_cooperate':   cost_cooperate,
                                    'cost_support':     cost_support,
                                    'cost_punish':      cost_punish,
                                    'power_social':     power_social,
                                    'punish_size':      punish_size,
                                    'alpha':            alpha,
                                    'epsilon':          epsilon,
                                }

                                filename = paramfilename(parameter)
                                f = open('./parameter/' + filename + '.yml', 'w')
                                f.write(yaml.dump(parameter, default_flow_style=False))
                                f.close()

if __name__ == "__main__":
    make_parameter_file()