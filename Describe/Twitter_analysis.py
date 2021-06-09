# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import seaborn
seaborn.set()
import matplotlib.pyplot as plt


def plot_time_distribution():
    df = np.load('../Dataset/TwitterEvent2012.npy', allow_pickle=True)
    data = pd.DataFrame(data=df, columns=["event_id", "tweet_id", "text", "created_at", "entity", "words"])

    result = []
    for name, group in data.groupby(["created_at"]):
        result.append([str(name), len(group)])
    result = pd.DataFrame(result, columns=['time', 'nums'])
    plt.figure()  # 画布
    plt.subplot()  # 子图
    plt.bar(x=result["time"], height=result["nums"], color='#66CDAA', alpha=0.6)
    plt.xticks(rotation=90, size=7)
    plt.xlabel('time')
    plt.ylabel('number of texts')
    plt.savefig("Tweets Time Distribution.png", bbox_inches='tight')
    plt.show()


def plot_event_distribution():
    df = np.load('../Dataset/TwitterEvent2012.npy', allow_pickle=True)
    data = pd.DataFrame(data=df, columns=["event_id", "tweet_id", "text", "created_at", "entity", "words"])
    result = []
    for name, group in data.groupby(["created_at"]):
        temp = group["event_id"].unique()
        result.append([str(name), len(temp)])
    result = pd.DataFrame(result, columns=['time', 'event_nums'])
    plt.figure()  # 画布
    plt.subplot()  # 子图
    plt.bar(x=result["time"], height=result["event_nums"], color='mediumpurple', alpha=0.6)
    plt.xticks(rotation=90, size=7)
    plt.xlabel('time')
    plt.ylabel('number of events')
    plt.savefig("Event Distribution.png", bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
    # plot_time_distribution()
    plot_event_distribution()