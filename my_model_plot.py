#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from my_model_config import *
from make_batch_file import paramfilename

def plot_line(df_for_line, datapath):
    fig = plt.figure(figsize=(12,9))
    plt.subplot(221)
    plt.plot(df_for_line['mean_gene_c'], color='r')
    plt.plot(df_for_line['median_gene_c'], color='b')
    plt.title('cooperate')
    plt.ylim(0, 1)
    plt.xlim(0, 10)

    plt.subplot(222)
    plt.plot(df_for_line['mean_gene_s'], color='r')
    plt.plot(df_for_line['median_gene_s'], color='b')
    plt.title('support')
    plt.ylim(0, 1)

    plt.subplot(223)
    plt.plot(df_for_line['mean_gene_pc'], color='r')
    plt.plot(df_for_line['median_gene_pc'], color='b')
    plt.title('punish noncooperater')
    plt.xlim(0, 10)
    plt.ylim(0, 1)

    plt.subplot(224)
    plt.plot(df_for_line['mean_gene_ps'], color='r')
    plt.plot(df_for_line['median_gene_ps'], color='b')
    plt.title('punish nonsupporter')
    plt.ylim(0, 1)
    
    filename = datapath + 'plot_img/my_result_average_line.png'
    plt.savefig(filename)

def plot_rate(m_gene, seed, datapath):
    for g_no in range(NUM_GROUPS):
        m_gene_g_no = m_gene.query('index == {group_number}'.format(group_number=g_no))
        m_gene_c = m_gene_g_no.loc[m_gene_g_no['gene_name'] == 'c']
        m_gene_s = m_gene_g_no.loc[m_gene_g_no['gene_name'] == 's']
        m_gene_c.reset_index(inplace=True, drop=True)
        m_gene_s.reset_index(inplace=True, drop=True)
        c_under_dot5 = (m_gene_c < 0.5).sum(axis=1)
        fig = plt.figure(figsize=(12,9))
        plt.plot(c_under_dot5)
        plt.savefig(datapath + 'plot_img/' + '{seed}_{group_number}.png'.format(seed=seed, group_number=g_no))
        plt.close()

def plot(datapath):
    filename_l = datapath + 'csv/leaders_gene_ave_seed={}.csv'
    filename_m = datapath + 'csv/groups_gene_ave_seed={}.csv'
    os.mkdir(datapath + 'plot_img')

    leaders = []
    members = []
    for i in range(S, MAX_REP):
        df_l = pd.read_csv(filename_l.format(i), index_col = 0, header = 0)
        df_m = pd.read_csv(filename_m.format(i), index_col = 0, header = 0)
        leaders.append(df_l.values)
        members.append(df_m.values)

    members = np.array(members)
    leaders = np.array(leaders)
    
    # 遺伝子値の世代変遷を見るための平均値と中央値の処理
    df_for_line = pd.DataFrame()
    df_for_line['mean_gene_c'] = np.mean(members[:, :, 0], axis=0)
    df_for_line['mean_gene_s'] = np.mean(members[:, :, 1], axis=0)
    df_for_line['mean_gene_pc'] = np.mean(leaders[:, :, 0], axis=0)
    df_for_line['mean_gene_ps'] = np.mean(leaders[:, :, 1], axis=0)
    df_for_line['median_gene_c'] = np.median(members[:, :, 0], axis=0)
    df_for_line['median_gene_s'] = np.median(members[:, :, 1], axis=0)
    df_for_line['median_gene_pc'] = np.median(leaders[:, :, 0], axis=0)
    df_for_line['median_gene_ps'] = np.median(leaders[:, :, 1], axis=0)
    plot_line(df_for_line, datapath)

    # 各グループの
    # filename = 'member_gene_seed={}.csv'
    # for seed in range(S, MAX_REP):
    #     m_gene = pd.read_csv(datapath + 'csv/' + filename.format(seed), index_col=0, header=0)
    #     plot_rate(m_gene, seed, datapath)

if __name__== "__main__":
    args = sys.argv
    rootpath = args[1]
    p_path_list = sorted(glob.glob('./parameter/*.yml'))
    for p_path in p_path_list:
        parameter = load_parameter(p_path)
        dirname = paramfilename(parameter)
        plot(rootpath + dirname + '/')
