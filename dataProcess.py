import numpy as np

def labelFeatures2RNN(users, reviews):
    for item in users:
        year = 2012
        view_seq = [[0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(124)]
        labels = [0 for i in range(109)]
        # print(item)
        if len(users[item]["reviews"]) == 0:
            continue
        for review in users[item]["reviews"]:
            view = reviews[review].copy()
            # print(view["date"])
            view["date"] = view["date"].split('-')

            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][0] += 1
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][1] += view["stars"]
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][2] += view["funny"]
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][3] += view["useful"]
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][4] += view["cool"]
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][5] += view["sentiment_score"]['neg']
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][6] += view["sentiment_score"]['neu']
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][7] += view["sentiment_score"]['pos']
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][8] += view["sentiment_vader"]
        for i in range(len(view_seq)):
            if view_seq[i][0] != 0:
                view_seq[i][1] /= view_seq[i][0]
                view_seq[i][2] /= view_seq[i][0]
                view_seq[i][3] /= view_seq[i][0]
                view_seq[i][4] /= view_seq[i][0]
                view_seq[i][5] /= view_seq[i][0]
                view_seq[i][6] /= view_seq[i][0]
                view_seq[i][7] /= view_seq[i][0]
                view_seq[i][8] /= view_seq[i][0]
        sign = 0
        for i in range(3, len(view_seq) - 12):
            sum_score = 0
            for score in view_seq[i + 1:i + 13]:
                sum_score += score[0]

            if view_seq[i][0] == 0:
                if sign == 0:
                    labels[i - 3] = 2
                else:
                    labels[i - 3] = 1

            elif view_seq[i][0] > 0:
                if sum_score > 0:
                    sign = 1
                    labels[i - 3] = 1
                else:
                    labels[i - 3] = 0
                    sign = 0

        users[item]["view_seq"] = view_seq
        users[item]["labels"] = labels
    friendChurn(users)

def labelFeatures2LR(users, reviews):
    for item in users:
        view_seq = [[0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(124)]
        labels = 0
        year = 2012
        # print(item)
        if len(users[item]["reviews"]) == 0:
            continue
        for review in users[item]["reviews"]:
            view = reviews[review].copy()
            # print(view["date"])
            view["date"] = view["date"].split('-')
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][0] += 1
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][1] += view["stars"]
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][2] += view["funny"]
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][3] += view["useful"]
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][4] += view["cool"]
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][5] += view["sentiment_score"]['neg']
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][6] += view["sentiment_score"]['neu']
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][7] += view["sentiment_score"]['pos']
            view_seq[(int(view["date"][0]) - year) * 12 + int(view["date"][1]) - 1][8] += view["sentiment_vader"]
        for i in range(len(view_seq)):
            if view_seq[i][0] != 0:
                view_seq[i][1] /= view_seq[i][0]
                view_seq[i][2] /= view_seq[i][0]
                view_seq[i][3] /= view_seq[i][0]
                view_seq[i][4] /= view_seq[i][0]
                view_seq[i][5] /= view_seq[i][0]
                view_seq[i][6] /= view_seq[i][0]
                view_seq[i][7] /= view_seq[i][0]
                view_seq[i][8] /= view_seq[i][0]

        sum_score = 0
        for i in range(len(view_seq) - 36, len(view_seq)):
            sum_score += view_seq[i][0]
        if sum_score > 0:
            labels = 1
            label_1 += 1
        users[item]["view_seq"] = view_seq
        users[item]["labels"] = labels
    print("staying user number: ", label_1)
    friendChurn(users)
    for item in users:
        features = np.array(users[item]["view_seq"])
        features = features[:88].sum(axis=0)
        users[item]['view_features'] = features
    return

def friendChurn(users):
    for item in users:
        label_churn = 0
        if len(users[item]['friends']) == 0:
            users[item]['friend_churn'] = 1
            continue
        for f in users[item]['friends']:
            if users[f]['labels'] == 0:
                label_churn += 1
        users[item]['friend_churn'] = label_churn / len(users[item]['friends'])
    return


def checkLabelFeatuers(users):
    for item in users:
        print(users[item]["view_seq"][:])
        for i in range(len(users[item]['view_seq'])):
            if users[item]['view_seq'][i][0] > 0:
                print(i)
        print(users[item]['labels'][:])
        # for i in range(len(users[item]['labels'])):
        # if users[item]['labels'][i] > 0:
        #   print(i)
        print(users[item]['since'])
        break
