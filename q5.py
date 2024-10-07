import pandas as pd
import numpy as np

user_item = pd.read_csv("user-item.csv", header=None)
item_data = pd.read_csv("item-data.csv", header=None)


def recommend_by_user(user):
    filtered_for_user = user_item.loc[user_item[0] == user]
    items_bought_by_which_users = user_item.groupby(
        1)[0].apply(list).reset_index()
    bought_items_set = set()

    for row in items_bought_by_which_users.itertuples():
        bought_items_set.add(row[1])

    def find_top_sims_for_an_item(item, count):
        output_items = []
        first_users = items_bought_by_which_users.loc[items_bought_by_which_users[1] == item][0].to_list()[
            0]
        for bought_item in bought_items_set:
            # print(bought_item, item)
            second_users = items_bought_by_which_users.loc[items_bought_by_which_users[1] == bought_item][0].to_list()[
                0]
            local_sim = cosine_sim(first_users, bought_item, second_users)
            if not bought_item == item:
                output_items.append((bought_item, item, local_sim))
        output_items.sort(key=lambda x: x[2], reverse=True)
        return output_items[:count]

    list_of_top_similarities = []
    for row in filtered_for_user.itertuples():
        local_list_of_tupples = find_top_sims_for_an_item(row[2], 5)
        for tu in local_list_of_tupples:
            list_of_top_similarities.append(tu)
    list_of_top_similarities.sort(key=lambda x: x[2], reverse=True)
    return list_of_top_similarities


def cosine_sim(first_users, b, second_users):

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


print(recommend_by_user(49722))
# دقت کنید این که تابع بهترین ها را بر نمیگرداند. بهترین های هر ایتم را بر میگرداند
# برای مثال اگر یک یوزر 3 ایتم خریده.
#  این تابع برای هر کدام از این ایتم ها 5 تا ایتم با بیشترین شباهت را بر میگرداند
