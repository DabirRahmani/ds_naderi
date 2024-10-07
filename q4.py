import pandas as pd
import numpy as np

user_item = pd.read_csv("user-item.csv", header=None)


def cosine_sim(a, b):
    user_bought_which_items = user_item.groupby(0)[1].apply(list).reset_index()
    items_bought_by_which_users = user_item.groupby(
        1)[0].apply(list).reset_index()

    user_bought_which_items = pd.DataFrame(user_bought_which_items)
    items_bought_by_which_users = pd.DataFrame(items_bought_by_which_users)

    first_users = items_bought_by_which_users.loc[items_bought_by_which_users[1] == a][0].to_list()[
        0]
    second_users = items_bought_by_which_users.loc[items_bought_by_which_users[1] == b][0].to_list()[
        0]

    all_users_of_first_and_second_list = list(set(first_users + second_users))

    df = pd.DataFrame(columns=all_users_of_first_and_second_list)

    df.loc[0] = np.zeros(len(all_users_of_first_and_second_list))

    df = pd.DataFrame(columns=all_users_of_first_and_second_list)
    df.loc[0] = np.zeros(len(all_users_of_first_and_second_list))
    for item in first_users:
        df.at[0, item] = 1
    df.loc[1] = np.zeros(len(all_users_of_first_and_second_list))
    for item in second_users:
        df.at[1, item] = 1

    return np.dot(df.loc[0].values, df.loc[1].values) / (np.linalg.norm(df.loc[0].values) * np.linalg.norm(df.loc[1].values))
