import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import config 
import make_batch_file

def plot(datapath):
    # 成員について
    df_l = []
    for i in range(1, config.MAX_REP):
        df = pd.read_csv('{}csv/members_q_seed={}.csv'.format(datapath, i), index_col='step', header=0).drop('Unnamed: 0', axis=1)
        df_l.append(df)

    cols = 5
    rows = 4
    fig = plt.figure(figsize=(15, 15))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    for i in range(0, len(df_l)):
        df_plot = df_l[i].groupby('step').mean()
        del df_plot['member_id']
        plt.subplot(cols, rows, i+1)
        
        plt.plot(df_plot['Qa_00'], label='Qa_00', alpha=0.6)
        plt.plot(df_plot['Qa_01'], label='Qa_01', alpha=0.6)
        plt.plot(df_plot['Qa_10'], label='Qa_10', alpha=0.6)
        plt.plot(df_plot['Qa_11'], label='Qa_11', alpha=0.6)
        plt.title('seed={}'.format(i+1))
        # plt.ylim(0, 40)
    plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0, 0, 0))
    plt.savefig(datapath + 'plot_img/members_dynamics.png')

    # 制裁者について
    df_l = []
    for i in range(1, config.MAX_REP):
        df = pd.read_csv('{}csv/leader_q_seed={}.csv'.format(datapath, i), index_col='step', header=0).drop('Unnamed: 0', axis=1)
        df_l.append(df)
    
    cols = 5
    rows = 4
    fig = plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    for i in range(0, len(df_l)):
        plt.subplot(cols, rows, i+1)
        plt.plot(df_l[i]['Qa_00'], label='Qa_00')
        plt.plot(df_l[i]['Qa_01'], label='Qa_01')
        plt.plot(df_l[i]['Qa_10'], label='Qa_10')
        plt.plot(df_l[i]['Qa_11'], label='Qa_11')
        plt.title('seed={}'.format(i+1))
        # plt.ylim(0, 3000)
    plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0, 0, 0))
    plt.savefig(datapath + 'plot_img/leader_dynamics.png')

if __name__== "__main__":
    args = sys.argv
    rootpath = args[1]
    p_path_list = sorted(glob.glob('./parameter/*.yml'))
    for p_path in p_path_list:
        parameter = config.load_parameter(p_path)
        dirname = make_batch_file.paramfilename(parameter)
        plot_img_dir = rootpath + dirname + '/plot_img'
        if not os.path.isdir(plot_img_dir):
            os.mkdir(plot_img_dir)
        plot(rootpath + dirname + '/')