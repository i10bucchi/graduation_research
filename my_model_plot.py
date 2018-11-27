#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from my_model_config import *

def plot_line(df_for_line):
    fig = plt.figure(figsize=(12,9))
    plt.subplot(221)
    plt.plot(df_for_line['gene_c'], color='y')
    plt.title('cooperate')
    plt.ylim(0, 1)

    plt.subplot(222)
    plt.plot(df_for_line['gene_s'], color='y')
    plt.title('support')
    plt.ylim(0, 1)

    plt.subplot(223)
    plt.plot(df_for_line['gene_pc'], color='y')
    plt.title('punish noncooperater')
    plt.ylim(0, 1)

    plt.subplot(224)
    plt.plot(df_for_line['gene_ps'], color='y')
    plt.title('punish nonsupporter')
    plt.ylim(0, 1)
    
    filename = 'my_result_average_line.png'
    plt.savefig(filename)

def main():
    df_for_line = pd.DataFrame()
    leaders = []
    members = []
    filename_l = 'leaders_gene_ave_seed={}.csv'
    filename_m = 'groups_gene_ave_seed={}.csv'
    for i in range(MAX_REP):
        df_l = pd.read_csv(filename_l.format(i), index_col = 0, header = 0)
        df_m = pd.read_csv(filename_m.format(i), index_col = 0, header = 0)
        leaders.append(df_l.values)
        members.append(df_m.values)
    
    df_for_line['gene_c'] = np.mean(np.array(members)[:, :, 0], axis=0)
    df_for_line['gene_s'] = np.mean(np.array(members)[:, :, 1], axis=0)
    df_for_line['gene_pc'] = np.mean(np.array(leaders)[:, :, 0], axis=0)
    df_for_line['gene_ps'] = np.mean(np.array(leaders)[:, :, 1], axis=0)
    print(np.array(members)[:, :, 0].shape)
    print(df_for_line)
    plot_line(df_for_line)

if __name__== "__main__":
    main()