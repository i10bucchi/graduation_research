#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from config import *
from make_batch_file import paramfilename

def plot_line(df_for_line, datapath):
    fig = plt.figure(figsize=(12,9))
    plt.subplot(221)
    plt.plot(df_for_line['mean_gene_c'], color='r', label="mean")
    plt.plot(df_for_line['median_gene_c'], color='b', label="median")
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

    plt.legend()
    
    filename = datapath + 'plot_img/my_result_average_line.png'
    plt.savefig(filename)

def plot_line_all(members, leaders, datapath):
    width = 3
    height_m = members.shape[0] / width + 1
    height_l = leaders.shape[0] / width + 1

    w_hight = MAX_REP - S * 3 / 5
    w_width = w_hight * 3 / 4 

    fig = plt.figure(figsize=(w_hight, w_width))
    for i, member in enumerate(members):
        plt.subplot(height_m, width, i+1)
        plt.ylim(0, 1)
        plt.plot(member[:, 0])
        plt.title('seed_{}'.format(S+i))
        plt.xlabel("Generation")
        plt.ylabel("Gene Value")
    plt.tight_layout()

    plt.savefig(datapath + 'plot_img/c_all.png')
    plt.close()

    fig = plt.figure(figsize=(w_hight, w_width))
    for i, member in enumerate(members):
        plt.subplot(height_m, width, i+1)
        plt.ylim(0, 1)
        plt.plot(member[:, 1])
        plt.title('seed_{}'.format(S+i))
        plt.xlabel("Generation")
        plt.ylabel("Gene Value")
    plt.tight_layout()

    plt.savefig(datapath + 'plot_img/s_all.png')
    plt.close()

    fig = plt.figure(figsize=(w_hight, w_width))
    for i, leader in enumerate(leaders):
        plt.subplot(height_l, width, i+1)
        plt.ylim(0, 1)
        plt.plot(leader[:, 0])
        plt.title('seed_{}'.format(S+i))
        plt.xlabel("Generation")
        plt.ylabel("Gene Value")
    plt.tight_layout()

    plt.savefig(datapath + 'plot_img/pc_all.png')
    plt.close()

    fig = plt.figure(figsize=(w_hight, w_width))
    for i, leader in enumerate(leaders):
        plt.subplot(height_l, width, i+1)
        plt.ylim(0, 1)
        plt.plot(leader[:, 1])
        plt.title('seed_{}'.format(S+i))
        plt.xlabel("Generation")
        plt.ylabel("Gene Value")
    plt.tight_layout()

    filename = datapath + 'plot_img/ps_all.png'
    plt.savefig(filename)
    plt.close()


def plot(datapath):
    filename_l = datapath + 'csv/leaders_gene_ave_seed={}.csv'
    filename_m = datapath + 'csv/groups_gene_ave_seed={}.csv'
    leaders = []
    members = []
    for i in range(S, MAX_REP):
        df_l = pd.read_csv(filename_l.format(i), index_col = 0, header = 0)
        df_m = pd.read_csv(filename_m.format(i), index_col = 0, header = 0)
        leaders.append(df_l.values)
        members.append(df_m.values)

    members = np.array(members)
    leaders = np.array(leaders)
    plot_line_all(members, leaders, datapath)
    
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

def plot_free(datapath):
    filename_l = datapath + 'csv/leader_gene_seed={}_generation={}.csv'
    filename_m = datapath + 'csv/member_gene_seed={}_generation={}.csv'

    dfm_list = []
    dfl_list = []
    for seed in range(S, MAX_REP):
        for i in range(10):
            g = (i * 50000) + 49999
            df = pd.read_csv(filename_m.format(seed, g), index_col=0, header=0)
            dfm_list.append(df)
            df = pd.read_csv(filename_l.format(seed, g), index_col=0, header=0)
            dfl_list.append(df)
            
        dfm = pd.concat(dfm_list)
        dfl = pd.concat(dfl_list)
        
        fig = plt.figure(figsize=(27,36))
        for g_no in range(20):
            # グループの抜き出し
            m_gene_g_no = dfm.loc[g_no]
            l_gene_g_no = dfl.loc[g_no]

            # 遺伝子の抜き出し
            m_gene_c = m_gene_g_no.loc[m_gene_g_no['gene_name'] == 'c']
            m_gene_s = m_gene_g_no.loc[m_gene_g_no['gene_name'] == 's']
            l_gene_pc = l_gene_g_no.loc[l_gene_g_no['gene_name'] == 'pc']
            l_gene_ps = l_gene_g_no.loc[l_gene_g_no['gene_name'] == 'ps']

            # indexの振り直し
            m_gene_c.reset_index(inplace=True, drop=True)
            m_gene_s.reset_index(inplace=True, drop=True)
            l_gene_pc.reset_index(inplace=True, drop=True)
            l_gene_ps.reset_index(inplace=True, drop=True)

            # 0.5以下の人数抜き出し
            c_under_dot3   = (m_gene_c <= 0.3).sum(axis=1)
            c_dot3_to_dot7 = ( (m_gene_c > 0.3) & (m_gene_c < 0.7) ).sum(axis=1)
            c_over_dot7    = (m_gene_c >= 0.7).sum(axis=1)

            plt.subplot(20, 2, g_no*2+1)
            plt.stackplot(c_under_dot3.index, [c_under_dot3.values, c_dot3_to_dot7.values, c_over_dot7.values], colors=["lightskyblue", "lightgreen", "tomato"])
            plt.legend(loc='upper left')

            plt.title("Num Of Peaple Separated by Gean Values Group-{}".format(g_no))
            plt.ylim(-1, 21)
            plt.xlabel("Generation")
            plt.ylabel("Num Of People")

            plt.subplot(20, 2, g_no*2+2)
            plt.plot(l_gene_pc.iloc[:, 0], color='b')
            plt.title("PC Gene Values Group-{}".format(g_no))
            plt.ylim(-0.1, 1.1)
            plt.xlabel("Generation")
            plt.ylabel("Gene Values")
        
        plt.tight_layout()
        
        plt.savefig(datapath + 'plot_img/freeplot_seed={}.png'.format(seed))
    plt.close()

if __name__== "__main__":
    args = sys.argv
    rootpath = args[1]
    p_path_list = sorted(glob.glob('./parameter/*.yml'))
    for p_path in p_path_list:
        parameter = load_parameter(p_path)
        dirname = paramfilename(parameter)
        plot_img_dir = rootpath + dirname + '/plot_img'
        if not os.path.isdir(plot_img_dir):
            os.mkdir(plot_img_dir)
        plot(rootpath + dirname + '/')
        # plot_free(rootpath + dirname + '/')