import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import config 
import make_batch_file

def plot_digest(rootpath, datapaths):
    '''
    abstract:
        指定した全てのパラメーターの実験結果比較を平均値を使って行う
    input:
        datapaths:    list
            各実験結果のcsvが格納されている場所へのパス
    output:
        ---
    '''
    df_m_folders = []
    df_l_folders = []
    
    for datapath in datapaths:
        df_m_list = []
        df_l_list = []
        for i in range(1, config.MAX_REP):
            df_m = pd.read_csv('{}/csv/members_q_seed={}.csv'.format(datapath, i), index_col='step', header=0).drop('Unnamed: 0', axis=1)
            df_l = pd.read_csv('{}/csv/leader_q_seed={}.csv'.format(datapath, i), index_col='step', header=0).drop('Unnamed: 0', axis=1)
            df_m['rep'] = i
            df_l['rep'] = i
            df_m_list.append(df_m)
            df_l_list.append(df_l)
        df_m_folders.append( pd.concat(df_m_list) )
        df_l_folders.append( pd.concat(df_l_list) )
    
    if ( len(datapaths) >= 1 ) and ( len(datapaths) <= 4 ):
        cols = len(datapaths)
        rows = 1
        
    else:
        cols = 4
        rows = int(len(datapaths) / 4) + 1

    fig = plt.figure(figsize=(15, 20))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for i in range(0, len(df_m_folders)):
        df_plot = df_m_folders[i].groupby(['step']).mean()
        plt.subplot(rows, cols, i+1)
        
        plt.plot(df_plot['Qa_00'], label='Qa_00', alpha=0.6)
        plt.plot(df_plot['Qa_01'], label='Qa_01', alpha=0.6)
        plt.plot(df_plot['Qa_10'], label='Qa_10', alpha=0.6)
        plt.plot(df_plot['Qa_11'], label='Qa_11', alpha=0.6)
        plt.title(datapaths[i].split('/')[2])
    plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0, 0, 0))
    plt.savefig('{}/result_digest_members.png'.format(rootpath))

    fig = plt.figure(figsize=(15, 20))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for i in range(0, len(df_l_folders)):
        df_plot = df_l_folders[i].groupby(['step']).mean()
        plt.subplot(rows, cols, i+1)
        
        plt.plot(df_plot['Qa_00'], label='Qa_00', alpha=0.6)
        plt.plot(df_plot['Qa_01'], label='Qa_01', alpha=0.6)
        plt.plot(df_plot['Qa_10'], label='Qa_10', alpha=0.6)
        plt.plot(df_plot['Qa_11'], label='Qa_11', alpha=0.6)
        plt.title(datapaths[i].split('/')[2])
    plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0, 0, 0))
    plt.savefig('{}/result_digest_leaders.png'.format(rootpath))

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
        plt.subplot(rows, cols, i+1)
        
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
        plt.subplot(rows, cols, i+1)
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
    datapaths = []
    for p_path in p_path_list:
        parameter = config.load_parameter(p_path)
        dirname = make_batch_file.paramfilename(parameter)
        plot_img_dir = rootpath + dirname + '/plot_img'
        if not os.path.isdir(plot_img_dir):
            os.mkdir(plot_img_dir)
        plot(rootpath + dirname + '/')
        datapaths.append(rootpath + dirname + '/')
    plot_digest(rootpath, datapaths)