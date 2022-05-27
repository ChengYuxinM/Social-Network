import matplotlib.pyplot as plt
import numpy as np


def FeaturesProcess2RNN(users, reviews):
    seq = {}
    for item in users:
        user_features = np.array([])
        # user_features = np.append(user_features, users[item]['review_count'])
        user_features = np.append(user_features, users[item]['fans'])
        user_features = np.append(user_features, len(users[item]['friends']))
        user_features = np.append(user_features, users[item]['comments'])
        user_features = np.append(user_features, users[item]['compliment'])
        user_features = np.append(user_features, users[item]['stars'])
        user_features = np.append(user_features, users[item]['friend_churn'])
        user_features = np.append(user_features, users[item]['review_count_valid'])



        view_featuers = np.zeros(shape=(0, 37))
        for i in range(len(users[item]['labels'])):
            if users[item]['labels'][i] == 2:
                continue
            features = np.array([])
            for j in range(4):
                features = np.append(features, users[item]['view_seq'][i + j])
            features = np.append(features, users[item]['labels'][i])
            view_featuers = np.vstack((view_featuers, features))

        user_features = np.tile(user_features, (view_featuers.shape[0], 1))
        user_features = np.hstack((user_features, view_featuers))
        seq[item] = user_features

    return seq


def FeaturesProcess2LR(users, reviews):
    seq = {}
    for item in users:
        user_features = np.array([])
        user_features = np.append(user_features, users[item]['review_count'])
        user_features = np.append(user_features, users[item]['fans'])
        user_features = np.append(user_features, len(users[item]['friends']))
        user_features = np.append(user_features, users[item]['comments'])
        user_features = np.append(user_features, users[item]['compliment'])
        user_features = np.append(user_features, users[item]['stars'])
        user_features = np.append(user_features, users[item]['friend_churn'])

        user_features = np.hstack((user_features, users[item]['view_features']))
        user_features = np.hstack((user_features, users[item]['labels']))
        seq[item] = user_features
    return seq


def statistic_visualization(seq):
    count_0 = np.array([])
    count_1 = np.array([])
    stars_0 = np.array([])
    stars_1 = np.array([])
    funny_0 = np.array([])
    funny_1 = np.array([])
    useful_0 = np.array([])
    useful_1 = np.array([])
    cool_0 = np.array([])
    cool_1 = np.array([])
    neg_0 = np.array([])
    neg_1 = np.array([])
    neu_0 = np.array([])
    neu_1 = np.array([])
    pos_0 = np.array([])
    pos_1 = np.array([])
    com_0 = np.array([])
    com_1 = np.array([])

    for item in seq:
        for view in seq[item]:
            count = 0
            stars = 0
            funny = 0
            useful = 0
            cool = 0
            neg = 0
            neu = 0
            pos = 0
            com = 0
            last_count = 0
            last_stars = 0
            last_funny = 0
            last_useful = 0
            last_cool = 0
            if view[-1] == -1:
                for i in range(4):
                    count += view[13 + 9 * i]
                    stars += view[13 + 9 * i + 1]
                    funny += view[13 + 9 * i + 2]
                    useful += view[13 + 9 * i + 3]
                    cool += view[13 + 9 * i + 4]
                    neg += view[13 + 9 * i + 5]
                    neu += view[13 + 9 * i + 6]
                    pos += view[13 + 9 * i + 7]
                    com += view[13 + 9 * i + 8]
                count_0 = np.append(count_0, count / 4)
                stars_0 = np.append(stars_0, stars / 4)
                funny_0 = np.append(funny_0, funny / 4)
                useful_0 = np.append(useful_0, useful / 4)
                cool_0 = np.append(cool_0, cool / 4)
                neg_0 = np.append(neg_0, neg / 4)
                neu_0 = np.append(neu_0, neu / 4)
                pos_0 = np.append(pos_0, pos / 4)
                com_0 = np.append(com_0, com / 4)
            else:
                for i in range(4):
                    count += view[13 + 9 * i]
                    stars += view[13 + 9 * i + 1]
                    funny += view[13 + 9 * i + 2]
                    useful += view[13 + 9 * i + 3]
                    cool += view[13 + 9 * i + 4]
                    neg += view[13 + 9 * i + 5]
                    neu += view[13 + 9 * i + 6]
                    pos += view[13 + 9 * i + 7]
                    com += view[13 + 9 * i + 8]
                count_1 = np.append(count_1, count / 4)
                stars_1 = np.append(stars_1, stars / 4)
                funny_1 = np.append(funny_1, funny / 4)
                useful_1 = np.append(useful_1, useful / 4)
                cool_1 = np.append(cool_1, cool / 4)
                neg_1 = np.append(neg_1, neg / 4)
                neu_1 = np.append(neu_1, neu / 4)
                pos_1 = np.append(pos_1, pos / 4)
                com_1 = np.append(com_1, com / 4)
        # break
    # print(count_0)
    # print(count_1)

    plt.figure(figsize=(20, 80))
    plt.subplot(9, 2, 1)
    min = count_0.min()
    max = count_0.max()
    plt.hist(count_0, bins=int(max - min + 1), facecolor="blue", edgecolor="black", alpha=0.7)
    min = count_1.min()
    max = count_1.max()
    plt.subplot(9, 2, 2)
    plt.hist(count_1, bins=int(max - min + 1), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 3)
    min = stars_0.min()
    max = stars_0.max()
    plt.hist(stars_0, bins=int(max - min + 1), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 4)
    min = stars_1.min()
    max = stars_1.max()
    plt.hist(stars_1, bins=int(max - min + 1), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 5)
    min = funny_0.min()
    max = funny_0.max()
    plt.hist(funny_0, bins=int(max - min + 1), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 6)
    min = funny_1.min()
    max = funny_1.max()
    plt.hist(funny_1, bins=int(max - min + 1), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 7)
    min = useful_0.min()
    max = useful_0.max()
    plt.hist(useful_0, bins=int(max - min + 1), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 8)
    min = useful_1.min()
    max = useful_1.max()
    plt.hist(useful_1, bins=int(max - min + 1), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 9)
    min = cool_0.min()
    max = cool_0.max()
    plt.hist(cool_0, bins=int(max - min + 1), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 10)
    min = cool_1.min()
    max = cool_1.max()
    plt.hist(cool_1, bins=int(max - min + 1), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 11)
    min = neg_0.min()
    max = neg_0.max()
    plt.hist(neg_0, bins=int((max - min + 1) / 0.05), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 12)
    min = neg_1.min()
    max = neg_1.max()
    plt.hist(neg_1, bins=int((max - min + 1) / 0.05), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 13)
    min = neu_0.min()
    max = neu_0.max()
    plt.hist(neu_0, bins=int((max - min + 1) / 0.05), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 14)
    min = neu_1.min()
    max = neu_1.max()
    plt.hist(neu_1, bins=int((max - min + 1) / 0.05), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 15)
    min = pos_0.min()
    max = pos_0.max()
    plt.hist(pos_0, bins=int((max - min + 1) / 0.05), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 16)
    min = pos_1.min()
    max = pos_1.max()
    plt.hist(pos_1, bins=int((max - min + 1) / 0.05), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 17)
    min = com_0.min()
    max = com_0.max()
    plt.hist(com_0, bins=int((max - min + 1) / 0.05), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.subplot(9, 2, 18)
    min = com_1.min()
    max = com_1.max()
    plt.hist(com_1, bins=int((max - min + 1) / 0.05), facecolor="blue", edgecolor="black", alpha=0.7)

    plt.savefig('./hist.png')
    plt.show()
    print(neg_0)
    print(neg_1)
    return count_0, count_1