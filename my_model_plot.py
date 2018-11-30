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

# ヒストグラムのための分割世代倍数
NUM_GRID = 5
G_GRID = 200000

def plot_line(df_for_line, datapath):
    fig = plt.figure(figsize=(12,9))
    plt.subplot(221)
    plt.plot(df_for_line['mean_gene_c'], color='r')
    plt.plot(df_for_line['median_gene_c'], color='b')
    plt.title('cooperate')
    plt.ylim(0, 1)

    plt.subplot(222)
    plt.plot(df_for_line['mean_gene_s'], color='r')
    plt.plot(df_for_line['median_gene_s'], color='b')
    plt.title('support')
    plt.ylim(0, 1)

    plt.subplot(223)
    plt.plot(df_for_line['mean_gene_pc'], color='r')
    plt.plot(df_for_line['median_gene_pc'], color='b')
    plt.title('punish noncooperater')
    plt.ylim(0, 1)

    plt.subplot(224)
    plt.plot(df_for_line['mean_gene_ps'], color='r')
    plt.plot(df_for_line['median_gene_ps'], color='b')
    plt.title('punish nonsupporter')
    plt.ylim(0, 1)
    
    filename = datapath + 'plot_img/my_result_average_line.png'
    plt.savefig(filename)

def plot_histgram(df_for_histgram, datapath):
    gene_name_list = ['c', 's', 'pc', 'ps']
    for gene_name in gene_name_list:
        fig = plt.figure(figsize=(12,4))
        for i in range(NUM_GRID):
            pos = 151 + i
            plt.subplot(pos)
            generation = G_GRID*i
            plt.hist(df_for_histgram['gene_{gene_name}_{generation}'.format(gene_name=gene_name, generation=generation)], bins=16, range=(0, 1))
            plt.title('generation = {generation}'.format(generation=generation))
            plt.xlim(0, 1)
            plt.ylim(0, MAX_REP)
        filename = datapath + 'plot_img/my_result_gene_{gene_name}_histgram.png'.format(gene_name=gene_name)
        plt.savefig(filename)

# def plot_rateline(df_for_rateline, datapath):
#     fig = plt.figure(figsize=(12,9))
#     plt.subplot(221)
#     plt.plot(df_for_rateline['gene_c_1'], color='r')
#     plt.plot(df_for_rateline['gene_c_2'], color='b')
#     plt.title('cooperate')
#     plt.ylim(0, 100)

#     plt.subplot(222)
#     plt.plot(df_for_rateline['gene_s_1'], color='r')
#     plt.plot(df_for_rateline['gene_s_2'], color='b')
#     plt.title('support')
#     plt.ylim(0, 100)

#     plt.subplot(223)
#     plt.plot(df_for_rateline['gene_pc_1'], color='r')
#     plt.plot(df_for_rateline['gene_pc_2'], color='b')
#     plt.title('punish noncooperater')
#     plt.ylim(0, 100)

#     plt.subplot(224)
#     plt.plot(df_for_rateline['gene_ps_1'], color='r')
#     plt.plot(df_for_rateline['gene_ps_2'], color='b')
#     plt.title('punish nonsupporter')
#     plt.ylim(0, 100)
    
#     filename = datapath + 'plot_img/my_result_rate_line.png'
#     plt.savefig(filename)

def plot(datapath):
    filename_l = datapath + 'csv/leaders_gene_ave_seed={}.csv'
    filename_m = datapath + 'csv/groups_gene_ave_seed={}.csv'
    os.mkdir(datapath + 'plot_img')

    leaders = []
    members = []
    for i in range(MAX_REP):
        df_l = pd.read_csv(filename_l.format(i), index_col = 0, header = 0)
        df_m = pd.read_csv(filename_m.format(i), index_col = 0, header = 0)
        leaders.append(df_l.values)
        members.append(df_m.values)

    members = np.array(members)
    leaders = np.array(leaders)
    
    # # 遺伝子値の世代変遷を見るための平均値と中央値の処理
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

    # 遺伝子値の分布を見るためのヒストグラムの処理
    df_for_histgram = pd.DataFrame()
    for i in range(NUM_GRID):
        df_for_histgram['gene_c_{generation}'.format(generation=G_GRID*i)] = members[:, G_GRID*i, 0]
        df_for_histgram['gene_s_{generation}'.format(generation=G_GRID*i)] = members[:, G_GRID*i, 1]
        df_for_histgram['gene_pc_{generation}'.format(generation=G_GRID*i)] = leaders[:, G_GRID*i, 0]
        df_for_histgram['gene_ps_{generation}'.format(generation=G_GRID*i)] = leaders[:, G_GRID*i, 1]
    plot_histgram(df_for_histgram, datapath)

    # # 遺伝子値の比率の背際変遷を見るための処理
    # df_for_rateline = pd.DataFrame()
    # df_for_rateline['gene_c_1'] = np.sum(members[:, :, 0] > 0.5, axis=0)
    # df_for_rateline['gene_c_2'] = np.sum(members[:, :, 0] <= 0.5, axis=0)
    # df_for_rateline['gene_s_1'] = np.sum(members[:, :, 1] > 0.5, axis=0)
    # df_for_rateline['gene_s_2'] = np.sum(members[:, :, 1] <= 0.5, axis=0)
    # df_for_rateline['gene_pc_1'] = np.sum(leaders[:, :, 0] > 0.5, axis=0)
    # df_for_rateline['gene_pc_2'] = np.sum(leaders[:, :, 0] <= 0.5, axis=0)
    # df_for_rateline['gene_ps_1'] = np.sum(leaders[:, :, 1] > 0.5, axis=0)
    # df_for_rateline['gene_ps_2'] = np.sum(leaders[:, :, 1] <= 0.5, axis=0)
    # plot_rateline(df_for_rateline, datapath)

if __name__== "__main__":
    args = sys.argv
    rootpath = args[1]
    p_path_list = sorted(glob.glob('./parameter/*.yml'))
    for p_path in p_path_list:
        parameter = load_parameter(p_path)
        dirname = paramfilename(parameter)
        plot(rootpath + dirname + '/')