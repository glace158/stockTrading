import os
import pandas as pd
import matplotlib.pyplot as plt
from common.fileManager import Config, File

def save_log_graph(log_path, save_path="./PPO_figs/",env_name = 'Richdog', fig_num = 0):
    print("============================================================================================")
    

    #fig_num = 0     #### change this to prevent overwriting figures in same env_name folder
    plot_avg = True    # plot average of all runs; else plot all runs separately
    fig_width = 10
    fig_height = 6

    # smooth out rewards to get a smooth and a less smooth (var) plot lines
    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson','gray', 'black']

    fig_directory = save_path + env_name + '/'

    fig_save_path =  'PPO_' + env_name + '_fig_' + str(fig_num) + '.png'

    fig_file = File(fig_directory , fig_save_path)

    all_runs = []
    print("loading data from : " + log_path)
    data = pd.read_csv(log_path)
    data = pd.DataFrame(data)
    print("data shape : ", data.shape)
    all_runs.append(data)
    print("--------------------------------------------------------------------------------------------")

    ax = plt.gca()

    if plot_avg:
        # average all runs
        df_concat = pd.concat(all_runs)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()

        # smooth out rewards to get a smooth and a less smooth (var) plot lines
        data_avg['reward_smooth'] = data_avg['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
        data_avg['reward_var'] = data_avg['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

        data_avg.plot(kind='line', x='timestep' , y='reward_smooth',ax=ax,color=colors[0],  linewidth=linewidth_smooth, alpha=alpha_smooth)
        data_avg.plot(kind='line', x='timestep' , y='reward_var',ax=ax,color=colors[0],  linewidth=linewidth_var, alpha=alpha_var)

        # keep only reward_smooth in the legend and rename it
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0]], ["reward_avg_" + str(len(all_runs)) + "_runs"], loc=2)

    else:
        for i, run in enumerate(all_runs):
            # smooth out rewards to get a smooth and a less smooth (var) plot lines
            run['reward_smooth_' + str(i)] = run['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
            run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

            # plot the lines
            run.plot(kind='line', x='timestep' , y='reward_smooth_' + str(i),ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_smooth, alpha=alpha_smooth)
            run.plot(kind='line', x='timestep' , y='reward_var_' + str(i),ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var)

        # keep alternate elements (reward_smooth_i) in the legend
        handles, labels = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        for i in range(len(handles)):
            if(i%2 == 0):
                new_handles.append(handles[i])
                new_labels.append(labels[i])
        ax.legend(new_handles, new_labels, loc=2)

    # ax.set_yticks(np.arange(0, 1800, 200))
    # ax.set_xticks(np.arange(0, int(4e6), int(5e5)))

    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Rewards", fontsize=12)

    plt.title(env_name, fontsize=14)

    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)

    print("============================================================================================")
    plt.savefig(fig_file.get_file_path())
    print("figure saved at : ", fig_file.get_file_path())
    print("============================================================================================")

    #plt.show()
    return fig_file.get_file_path()

# 행동 그래프 그리기
def save_action_graph(action_path, save_path ="./Data_graph/" ,env_name = 'Richdog', fig_num = 0):

    save_path = save_path + env_name + '/'
    file_name =  '/PPO_' + env_name + '_action_fig_' + str(fig_num) + '.png'
    
    data = pd.read_csv(action_path)

    # 그래프 그리기
    fig, ax1 = plt.subplots(figsize=(60, 10))

    # Reward 막대 그래프 (아래쪽 Y축 공유)
    ax1.spines['right'].set_position(('outward', 60))  # 추가 Y축을 오른쪽으로 이동

    reward_colors = ['red' if r < 0 else 'blue' for r in data['reward']]  # 음수는 빨간색, 양수는 파란색
    ax1.bar(data['timestep'], data['reward'], width=0.4, alpha=0.8, label='Reward', color=reward_colors)
    ax1.tick_params(axis='x', rotation=90, labelsize=8)

    # y축 0에 선 추가
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, label='y = 0 Line')

    ax1.set_ylabel('Reward and Daily Rate')
    ax1.legend(loc='lower center')
    
    # Current Amount 선 그래프 (오른쪽 Y축)
    ax2 = ax1.twinx()  # 오른쪽 Y축 추가
    ax2.plot(data['timestep'], data['total_amt'], label='Total Amount', color='Orange')
    ax2.set_ylabel('Total Amount')
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='x', rotation=90, labelsize=8)
    ax2.axhline(y=data['total_amt'][0], color='Orange', linestyle='--', linewidth=1, label='init Amount Line')
    
    # Price 선 그래프 (왼쪽 Y축)
    ax3 = ax1.twinx()  # 새로운 Y축 추가
    ax3.plot(data['timestep'], data['price'], label='Price', color='black')
    
    for index, row in data.iterrows():
        if row['order_qty'] == 0:  # order_qty가 0인 경우 회색 점
            ax3.scatter(row['timestep'], row['price'], color='gray', label='Order Qty = 0' if index == 0 else "")
            ax3.annotate(f"{int(row['order_qty'])}", (row['timestep'], row['price']), textcoords="offset points", xytext=(-10, -10), color='gray')
        elif row['action'] > 0:  # action이 음수인 경우 빨간 점
            ax3.scatter(row['timestep'], row['price'], color='red', label='Action > 0' if index == 0 else "")
            ax3.annotate(f"{int(row['order_qty'])}", (row['timestep'], row['price']), textcoords="offset points", xytext=(-10, -10), color='red')
        elif row['action'] < 0:  # action이 양수인 경우 파란 점
            ax3.scatter(row['timestep'], row['price'], color='blue', label='Action < 0' if index == 0 else "")
            ax3.annotate(f"{int(row['order_qty'])}", (row['timestep'], row['price']), textcoords="offset points", xytext=(-10, 10), color='blue')
    
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Price')
    ax3.set_xticks(data['timestep'])  # x축 간격을 timestep에 맞춤
    ax3.tick_params(axis='x', rotation=90, labelsize=8)
    ax3.legend(loc='upper left')
    ax3.grid(True)

    # 그래프 제목
    plt.title('Action Graph')
    
    plt.savefig(save_path + file_name)

    return save_path + file_name
if __name__ == '__main__':

    save_log_graph()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
