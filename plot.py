import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import config 
import make_batch_file

def plot(datapath):
    df_q_m = []
    df_q_l = []
    df_r = []

    for i in range(1, config.MAX_REP):
        tmp = pd.read_csv('{}/csv/players_qrm_seed={}.csv'.format(datapath, i), header=0).rename(columns={'Unnamed: 0':'step'})
        df_q_m.append(tmp)
        tmp = pd.read_csv('{}/csv/players_qrl_seed={}.csv'.format(datapath, i), header=0).rename(columns={'Unnamed: 0':'step'})
        df_q_l.append(tmp)
        tmp = pd.read_csv('{}/csv/players_reward_seed={}.csv'.format(datapath, i), header=0).rename(columns={'Unnamed: 0':'step'})
        df_r.append(tmp)


    figure = plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    df_q_plot = pd.concat(df_q_m).groupby('step').mean()
    plt.plot(df_q_plot['Qr_00'], label='$dictatorship, online$', alpha=0.7)
    plt.plot(df_q_plot['Qr_01'], label='$dictatorship, batch$', alpha=0.7)
    plt.plot(df_q_plot['Qr_10'], label='$democracy,\:\,  online$', alpha=0.7)
    plt.plot(df_q_plot['Qr_11'], label='$democracy,\:\,  batch$', alpha=0.7)

    plt.title('Q value for game rule', fontsize=25)
    plt.xlabel('t', fontsize=20)
    plt.ylabel('$Qr$', fontsize=20)
    plt.tick_params(labelsize=20)

    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=20)


    plt.subplot(2, 1, 2)
    df_q_plot = pd.concat(df_q_l).groupby('step').mean()
    plt.plot(df_q_plot['Qr_00'], label='$dictatorship, online$', alpha=0.7)
    plt.plot(df_q_plot['Qr_01'], label='$dictatorship, batch$', alpha=0.7)
    plt.plot(df_q_plot['Qr_10'], label='$democracy,\:\,  online$', alpha=0.7)
    plt.plot(df_q_plot['Qr_11'], label='$democracy,\:\,  batch$', alpha=0.7)

    plt.title('Q value for game rule', fontsize=25)
    plt.xlabel('t', fontsize=20)
    plt.ylabel('$Qr$', fontsize=20)
    plt.tick_params(labelsize=20)

    plt.grid()

    plt.savefig(datapath + 'plot_img/players_q_value.png')

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