import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import config 
import make_batch_file

def plot(path):

    df_cn_l = []
    df_cr_l = []
    df_qa_l = []
    df_qap_l = []
    df_role_l = []

    for i in range(config.S, config.MAX_REP):
        tmp = pd.read_csv(path + 'csv/comunity_population_seed={}.csv'.format(i), header=0).rename(columns={'Unnamed: 0': 'step'})
        tmp['seed'] = i
        df_cn_l.append(tmp)
        tmp = pd.read_csv(path + 'csv/comunity_reward_seed={}.csv'.format(i), header=0).rename(columns={'Unnamed: 0': 'step'})
        tmp['seed'] = i
        df_cr_l.append(tmp)
        tmp = pd.read_csv(path + 'csv/players_qa_seed={}.csv'.format(i), header=0).rename(columns={'Unnamed: 0': 'step'})
        tmp['seed'] = i
        df_qa_l.append(tmp)
        tmp = pd.read_csv(path + 'csv/players_qap_seed={}.csv'.format(i), header=0).rename(columns={'Unnamed: 0': 'step'})
        tmp['seed'] = i
        df_qap_l.append(tmp)
        tmp = pd.read_csv(path + 'csv/role_seed={}.csv'.format(i), header=0).rename(columns={'Unnamed: 0': 'step'})
        tmp['seed'] = i
        df_role_l.append(tmp)
        
    df_cn = pd.concat(df_cn_l)
    df_cr = pd.concat(df_cr_l)
    df_qa = pd.concat(df_qa_l)
    df_qap = pd.concat(df_qap_l)
    df_role = pd.concat(df_role_l)


    # 成員人数の遷移
    plt.figure(figsize=(16,8))
    plt.suptitle('all members num')

    plt.subplots_adjust(wspace=0.05)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])

    plt.subplot(gs[0])
    nd_plot = df_role.loc[:, '0':str(config.NUM_PLAYERS-1)].sum(axis=1).values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=33, alpha=0.3, orientation="horizontal")
    plt.ylim(int(config.NUM_PLAYERS*0.8), config.NUM_PLAYERS)

    plt.subplot(gs[1])
    df_role['member_num'] = df_role.loc[:, '0':str(config.NUM_PLAYERS-1)].sum(axis=1)
    df_plot = df_role.groupby('step')['member_num'].mean()
    df_std = df_role.groupby('step')['member_num'].std()

    plt.fill_between(df_plot.index, df_plot - df_std, df_plot + df_std, facecolor='y', alpha=0.3, label='mean-std ~ mean+std')
    plt.plot(df_plot, label='mean')
    plt.ylim(int(config.NUM_PLAYERS*0.8), config.NUM_PLAYERS)
    plt.yticks([])

    plt.legend()

    plt.savefig(path + 'plot_img/members_num.png', bbox_inches="tight")

    # Qaの遷移
    plt.figure(figsize=(16,8))
    plt.suptitle('memebrs Qa')

    plt.subplots_adjust(wspace=0.05)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])

    plt.subplot(gs[0])
    nd_plot = df_qa.loc[:, 'Qa_00'].values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=30, color='tab:blue', alpha=0.3, orientation="horizontal")
    nd_plot = df_qa.loc[:, 'Qa_01'].values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=30, color='tab:orange', alpha=0.3, orientation="horizontal")
    nd_plot = df_qa.loc[:, 'Qa_10'].values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=30, color='tab:green', alpha=0.3, orientation="horizontal")
    nd_plot = df_qa.loc[:, 'Qa_11'].values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=30, color='tab:red', alpha=0.3, orientation="horizontal")

    plt.ylim(-1, 19)

    plt.subplot(gs[1])
    df_std = df_qa.groupby('step').std()
    df_plot = df_qa.groupby('step').mean()

    plt.fill_between(df_plot.index, df_plot['Qa_00'] - df_std['Qa_00'], df_plot['Qa_00'] + df_std['Qa_00'], color='tab:blue', facecolor='y', alpha=0.3)
    plt.plot(df_plot['Qa_00'], color='tab:blue', label='D, nS')
    plt.fill_between(df_plot.index, df_plot['Qa_01'] - df_std['Qa_01'], df_plot['Qa_01'] + df_std['Qa_01'], color='tab:orange', facecolor='y', alpha=0.3)
    plt.plot(df_plot['Qa_01'], color='tab:orange', label='D, S')
    plt.fill_between(df_plot.index, df_plot['Qa_10'] - df_std['Qa_10'], df_plot['Qa_10'] + df_std['Qa_10'], color='tab:green', facecolor='y', alpha=0.3)
    plt.plot(df_plot['Qa_10'], color='tab:green', label='C, nS')
    plt.fill_between(df_plot.index, df_plot['Qa_11'] - df_std['Qa_11'], df_plot['Qa_11'] + df_std['Qa_11'], color='tab:red', facecolor='y', alpha=0.3)
    plt.plot(df_plot['Qa_11'], color='tab:red', label='C, S')

    plt.ylim(-1, 19)
    plt.yticks([])
    plt.legend()

    plt.savefig(path + 'plot_img/members_Qa.png', bbox_inches="tight")

    # Qapの遷移
    plt.figure(figsize=(16,8))
    plt.suptitle('punishers Qap')

    plt.subplots_adjust(wspace=0.05)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])

    plt.subplot(gs[0])
    nd_plot = df_qap.loc[:, 'Qap_00'].values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=30, color='tab:blue', alpha=0.3, orientation="horizontal")
    nd_plot = df_qap.loc[:, 'Qap_01'].values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=30, color='tab:orange', alpha=0.3, orientation="horizontal")
    nd_plot = df_qap.loc[:, 'Qap_10'].values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=30, color='tab:green', alpha=0.3, orientation="horizontal")
    nd_plot = df_qap.loc[:, 'Qap_11'].values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=30, color='tab:red', alpha=0.3, orientation="horizontal")

    plt.ylim(-1, 19)

    plt.subplot(gs[1])
    df_std = df_qap.groupby('step').std()
    df_plot = df_qap.groupby('step').mean()

    plt.fill_between(df_plot.index, df_plot['Qap_00'] - df_std['Qap_00'], df_plot['Qap_00'] + df_std['Qap_00'], color='tab:blue', facecolor='y', alpha=0.3)
    plt.plot(df_plot['Qap_00'], color='tab:blue', label='nPD, nPnS')
    plt.fill_between(df_plot.index, df_plot['Qap_01'] - df_std['Qap_01'], df_plot['Qap_01'] + df_std['Qap_01'], color='tab:orange', facecolor='y', alpha=0.3)
    plt.plot(df_plot['Qap_01'], color='tab:orange', label='nPD, PnS')
    plt.fill_between(df_plot.index, df_plot['Qap_10'] - df_std['Qap_10'], df_plot['Qap_10'] + df_std['Qap_10'], color='tab:green', facecolor='y', alpha=0.3)
    plt.plot(df_plot['Qap_10'], color='tab:green', label='PD,   nPnS')
    plt.fill_between(df_plot.index, df_plot['Qap_11'] - df_std['Qap_11'], df_plot['Qap_11'] + df_std['Qap_11'], color='tab:red', facecolor='y', alpha=0.3)
    plt.plot(df_plot['Qap_11'], color='tab:red', label='PD,   PnS')

    plt.ylim(-1, 19)
    plt.yticks([])

    plt.legend()

    plt.savefig(path + 'plot_img/punishers_Qap.png', bbox_inches="tight")

    # コミュニティ内の成員数の遷移
    plt.figure(figsize=(12, 8))
    plt.suptitle('num comunity members')

    plt.subplots_adjust(wspace=0.05)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])

    plt.subplot(gs[0])
    nd_plot = df_cn.loc[:, '0':str(config.NUM_PLAYERS-1)][df_role.loc[:, '0':str(config.NUM_PLAYERS-1)] == 0].values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=config.NUM_PLAYERS-1, alpha=0.3, orientation="horizontal")

    plt.ylim(-1, config.NUM_PLAYERS)

    plt.subplot(gs[1])
    df_plot= df_cn.loc[:, '0':str(config.NUM_PLAYERS-1)][df_role.loc[:, '0':str(config.NUM_PLAYERS-1)] == 0]
    df_plot['step'] = df_cn['step']
    df_std = df_plot.groupby('step').std().mean(axis=1)
    df_plot = df_plot.groupby('step').mean().mean(axis=1)

    plt.fill_between(df_plot.index, df_plot - df_std, df_plot + df_std, facecolor='y', alpha=0.3, label='mean-std ~ mean+std')
    plt.plot(df_plot, label='mean')

    plt.ylim(-1, config.NUM_PLAYERS)
    plt.yticks([])

    plt.legend()

    plt.savefig(path + 'plot_img/comunity_size.png', bbox_inches="tight")

    # コミュニティの評価遷移
    plt.figure(figsize=(16, 8))
    plt.suptitle('comunity reward')

    plt.subplots_adjust(wspace=0.05)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])

    plt.subplot(gs[0])
    nd_plot= df_cr.loc[:, '1':str(config.NUM_PLAYERS-1)][df_role.loc[:, '1':str(config.NUM_PLAYERS-1)] == 0].values.reshape(-1)
    weights = np.ones(nd_plot.shape[0]) / nd_plot.shape[0]
    plt.hist(nd_plot, weights=weights, bins=config.NUM_PLAYERS-1, color='tab:orange', alpha=0.3, orientation="horizontal")

    plt.ylim(-1, 18)

    plt.subplot(gs[1])
    df_plot= df_cr.loc[:, '1':str(config.NUM_PLAYERS-1)][df_role.loc[:, '1':str(config.NUM_PLAYERS-1)] == 0]
    df_plot['step'] = df_cr['step']
    df_plot = df_plot.groupby('step').mean().mean(axis=1)

    plt.plot(df_cr.loc[:, 'step':'0'].groupby('step').mean(), label='free comunity')
    plt.plot(df_plot, label='other coumnities')

    plt.ylim(-1, 18)
    plt.yticks([])
    plt.legend()

    plt.savefig(path + 'plot_img/comunity_reward.png', bbox_inches="tight")

    # 制裁者の経験回数の分布
    plt.figure(figsize=(12, 8))
    df_plot = (config.MAX_TERN - df_role.loc[:, '1':'seed'].groupby('seed').sum()).values.reshape(-1)
    plt.hist(df_plot, bins=200)
    plt.savefig(path + 'plot_img/punisher_experiment_distribution.png', bbox_inches="tight")

if __name__== "__main__":
    args = sys.argv
    rootpath = args[1]
    p_path_list = sorted(glob.glob('./parameter/*.yml'))
    datapaths = []
    for p_path in p_path_list:
        parameter = config.load_parameter(p_path)
        dirname = make_batch_file.paramfilename(parameter)
        plot_img_dir = rootpath + dirname + '/plot_img'
        if not os.path.isdir(plot_img_dir):
            os.mkdir(plot_img_dir)
        plot(rootpath + dirname + '/')
        datapaths.append(rootpath + dirname + '/')