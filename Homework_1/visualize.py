import os
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow
import seaborn as sns
from matplotlib import font_manager
import time

def load_dataset(folder_path):
    dfs = []
    for fname in os.listdir(folder_path):
        if fname == 'part-00000.parquet':
            fpath = os.path.join(folder_path, fname)
            # Process one file at a time
            df = pd.read_parquet(fpath)
            # Optional: Select only needed columns to save memory
            df = df[['age', 'income', 'gender', 'country', 'is_active', 'registration_date']]
            dfs.append(df)
            # Clear memory after each file
            del df

    return pd.concat(dfs, ignore_index=True)

def visualize_dataset(df, title_prefix, save_path):

    fig, axs = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'{title_prefix} Visualization', fontsize=20)

    sns.histplot(df['age'], bins=30, kde=True, color='violet', ax=axs[0, 0])
    axs[0, 0].set_title('Age Distribution Histogram', )
    axs[0, 0].set_xlabel('Age', )

    sns.boxplot(x=df['age'], color='red', ax=axs[0, 1])
    axs[0, 1].set_title('Age Distribution Boxplot', )
    axs[0, 1].set_xlabel('Age')

    sns.histplot(df['income'], bins=30, kde=True, color='skyblue', ax=axs[0, 2])
    axs[0, 2].set_title('Income Distribution Histogram', )
    axs[0, 2].set_xlabel('Income')


    gender_counts = df['gender'].value_counts()
    axs[1, 0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    axs[1, 0].set_title('Gender Distribution', )


    country_counts = df['country'].value_counts()
    axs[1, 1].pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', colors=sns.color_palette("muted"))
    axs[1, 1].set_title('Country Distribution', )


    active_counts = df['is_active'].value_counts()
    labels = ['Active User' if val else 'Non-active User' for val in active_counts.index]
    axs[1, 2].pie(active_counts, labels=labels, autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
    axs[1, 2].set_title('Active User Distribution', )


    df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
    reg_counts = df['registration_date'].dt.to_period('M').value_counts().sort_index()
    reg_counts.plot(kind='bar', color='slateblue', ax=axs[2, 0])
    axs[2, 0].set_title('Distribution of Registration Month', )
    axs[2, 0].set_xlabel('Month of Registration')
    axs[2, 0].set_ylabel('# of Users')


    axs[2, 1].axis('off')
    axs[2, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close()


path_10g = './10G_data_new'
path_30g = './30G_data_new'

# 10G 数据处理
start_10g = time.time()
df_10g = load_dataset(path_10g)
load_time_10g = time.time() - start_10g
print(f"10G time used：{load_time_10g:.2f} s")

visualize_dataset(df_10g, title_prefix='10G', save_path='visualization_30G.png')

vis_time_10g = time.time() - start_10g
print(f"Visualizing 10G time used：{vis_time_10g:.2f} s")

'''
# 30G 数据处理
start_30g = time.time()
df_30g = load_dataset(path_30g)
load_time_30g = time.time() - start_30g
print(f"加载 30G 数据耗时：{load_time_30g:.2f} 秒")

visualize_dataset(df_30g, title_prefix='30G', save_path='visualization_30G.png')

vis_time_30g = time.time() - start_30g
print(f"可视化 30G 数据耗时：{vis_time_30g:.2f} 秒")
'''