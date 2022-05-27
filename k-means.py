import json
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import random
import numpy as np

#read data
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#check the noise and outlier
def check_user(user):

    #if some feature of user is too high or too low , the user is considered as noise and outlier
    if len(user["friends"]) > 10000:
        return False
    elif user["fans"] > 2000:
        return False
    elif user["review_count"] > 6000:
        return False
    elif user["stars"] < 0.5:
        return False
    elif user["stars"] > 4.5:
        return False
    else:
        return True

if __name__ == '__main__':

    #json_file = open('yelp_academic_dataset_user.json')

    #read the radius file and user file
    radius = {}
    with open("average_radius.json", errors='ignore') as f:
        for line in f:
            j = json.loads(line)
            print(type(j))
            for k, v in j.items():
                radius[k] = v

    cnt = 0
    users = []
    churned = []
    user_pkl = load_obj("users_new")
    count = 0
    print(len(radius))

    #create a user list data structure
    for item in user_pkl:
        if count == 50:
            break

        if not check_user(user_pkl[item]):
            continue

        #single features
        tem = []
        tem.append(len(user_pkl[item]["friends"]))

        #multi feature
        #tem.append(len(user_pkl[item]["friends"]))
        #tem.append(int(user_pkl[item]["review_count"]))
        #tem.append(int(user_pkl[item]["fans"]))


        #combined features
        '''
        tem.append(len(user_pkl[item]["friends"]) + 0.4 * int(user_pkl[item]["fans"]))
        tem.append(float(user_pkl[item]["friend_churn"]))
        tem.append(len(user_pkl[item]["comments"]) + 0.5 * int(user_pkl[item]["review_count"]))
        tem.append(int(user_pkl[item]["stars"]))
        tem.append(radius[item])
        tem.append(len(user_pkl[item]["business_id"]))
        '''

        users.append(tem)
        churned.append(int(user_pkl[item]["labels"]))
        count += 1
        print(count)

    print(users)
    #with open("yelp_academic_dataset_user.json", errors='ignore') as f:
    '''
    with open("small_user.json", errors='ignore') as f:
        for line in f:
            j = json.loads(line)
            tem = []
            tem.append(int(len(j['friends'])))
            tem.append(int(j['review_count']))
            tem.append(int(j['fans']))
            churned.append(random.randint(0,1))
            users.append(tem)
            print(j['user_id'])
            print(j['name'])
    '''
    '''
    cnt2 = 0
    review_df = pd.DataFrame()
    user_interview = {}
    user_last_interview = {}
    #with open("yelp_academic_dataset_review.json", errors= 'ignore') as f :
    with open("small_review.json", errors='ignore') as f:
        for line in f:
            j = json.loads(line)
            time = j['date']
            print(time)
            print(type(time))
            #review.append(j)
            if user_interview.get(j['user_id']) == None:
                review_list = []
                review_list.append(j)
                user_interview[j['user_id']] = review_list
            else:
                user_interview[j['user_id']].append(j)
            #if user_last_interview.get(j['user_id']) == None:
            print(j['user_id'])
            cnt2 += 1
            print(cnt2)
    print(cnt)
    print(cnt2)
    print(len(users))
    print(len(user_interview))
    sum = 0
    for k, v in user_interview.items():
        #print(k)
        sum += len(v)
    print(sum)
    '''

    #K means ++ part
    print(len(users))
    model = cluster.KMeans(n_clusters=4, max_iter=30, init="k-means++", verbose = True)
    model.fit(users)
    predicted = model.predict(users)
    print('kmeans value', predicted)
    #'#96CCCB', '#D1C8DA', '#FEB2B4', '#9DC3E7'
    #'#32B897', '#8983BF', '#F27970', '#05B9E2'

    #T-SNE part
    cluster_colors = []
    for i in range(len(predicted)):
        if predicted[i] == 0:
            #cluster_colors.append('b')
            #cluster_colors.append('#96CCCB')
            cluster_colors.append('#32B897')
        elif predicted[i] == 1:
            #cluster_colors.append('y')
            #cluster_colors.append('#D1C8DA')
            cluster_colors.append('#8983BF')
        elif predicted[i] == 2:
            #cluster_colors.append('g')
            #cluster_colors.append('#FEB2B4')
            cluster_colors.append('#F27970')
        elif predicted[i] == 3:
            #cluster_colors.append('r')
            #cluster_colors.append('#9DC3E7')
            cluster_colors.append('#05B9E2')
    data_TSNE = TSNE(n_components=2, verbose = 1).fit_transform(users)
    x1_axis = data_TSNE[:, 0]
    x2_axis = data_TSNE[:, 1]
    plt.figure(figsize=(14, 7))
    plt.scatter(x1_axis, x2_axis, s = 1,c=cluster_colors)  # marker='s', s=100, cmap=plt.cm.Paired)
    plt.title("KMeans")
    plt.show()


    # the clustering result part
    #score = silhouette_score(users, predicted)
    cnt_0 = 0
    cnt_1 = 0
    cnt_2 = 0
    cnt_3 = 0
    #cnt_4 = 0
    churned0 = 0
    churned1 = 0
    churned2 = 0
    churned3 = 0
    #churned4 = 0
    cluster_0 = []
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []
    for i in range(len(predicted)):
        pre = predicted[i]
        if pre == 0:
            cnt_0 += 1
            if churned[i] == 0:
                churned0 += 1
            if len(cluster_0) == 10:
                continue
            else:
                cluster_0.append(users[i])

        elif pre == 1:
            cnt_1 += 1
            if churned[i] == 0:
                churned1 += 1
            if len(cluster_1) == 10:
                continue
            else:
                cluster_1.append(users[i])

        elif pre == 2:
            cnt_2 += 1
            if churned[i] == 0:
                churned2 += 1
            if len(cluster_2) == 10:
                continue
            else:
                cluster_2.append(users[i])

        elif pre == 3:
            cnt_3 += 1
            if churned[i] == 0:
                churned3 += 1
            if len(cluster_3) == 10:
                continue
            else:
                cluster_3.append(users[i])
        #elif pre == 4:
        #    cnt_4 += 1
        #    if churned[i] == 1:
        #        churned4 += 1

    print("cluster 0 ")
    print(cluster_0)
    print("max, min, ave")
    df_0 = pd.DataFrame(cluster_0)
    print(df_0.max())
    print(df_0.min())
    print(df_0.mean())
    print("cluster 1")
    print(cluster_1)
    print("max, min, ave")
    df_1 = pd.DataFrame(cluster_1)
    print(df_1.max())
    print(df_1.min())
    print(df_1.mean())
    print("cluster 2")
    print(cluster_2)
    print("max, min, ave")
    df_2 = pd.DataFrame(cluster_2)
    print(df_2.max())
    print(df_2.min())
    print(df_2.mean())
    print("cluster 3")
    print(cluster_3)
    print("max, min, ave")
    df_3 = pd.DataFrame(cluster_3)
    print(df_3.max())
    print(df_3.min())
    print(df_3.mean())
    #x = np.array([cnt_0, cnt_1, cnt_2, cnt_3, cnt_4])
    x = np.array([cnt_0, cnt_1, cnt_2, cnt_3])
    #plt.pie(x,autopct='%.f%%', labels= ["0", "1", "2", "3", "4"])
    #plt.pie(x, autopct='%.f%%', labels=["0", "1", "2", "3"], colors = ['b', 'y', 'g', 'r'])

    #plt.pie(x, autopct='%.f%%', labels=["0", "1", "2", "3"], colors=['#96CCCB', '#D1C8DA', '#FEB2B4', '#9DC3E7'])
    plt.pie(x, autopct='%.f%%', labels=["0", "1", "2", "3"], colors=['#32B897', '#8983BF', '#F27970', '#05B9E2'])
    plt.show()

    #x_data = ["0", "1", "2", "3", "4"]
    x_data = ["0", "1", "2", "3"]
    #y_data = [churned0 / cnt_0, churned1 / cnt_1, churned2 / cnt_2, churned3 / cnt_3, churned4 / cnt_4]
    y_data = [churned0 / cnt_0, churned1 / cnt_1, churned2 / cnt_2, churned3 / cnt_3]
    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    #total_churned = churned0 + churned1 + churned2 + churned3 + churned4
    total_churned = churned0 + churned1 + churned2 + churned3
    for i in range(len(x_data)):
        if i == 0:
            #plt.bar(x_data[i], y_data[i], color = 'b')
            #plt.bar(x_data[i], y_data[i], color='#96CCCB')
            plt.bar(x_data[i], y_data[i], color='#32B897')
        elif i == 1:
            #plt.bar(x_data[i], y_data[i], color = 'y')
            #plt.bar(x_data[i], y_data[i], color='#D1C8DA')
            plt.bar(x_data[i], y_data[i], color='#8983BF')
        elif i == 2:
            #plt.bar(x_data[i], y_data[i], color='g')
            #plt.bar(x_data[i], y_data[i], color='#FEB2B4')
            plt.bar(x_data[i], y_data[i], color='#F27970')
        elif i == 3:
            #plt.bar(x_data[i], y_data[i], color='r')
            #plt.bar(x_data[i], y_data[i], color='#9DC3E7')
            plt.bar(x_data[i], y_data[i], color='#05B9E2')

    plt.axhline(y = total_churned / len(users), ls='--', c='red')
    plt.title("churned ")
    plt.xlabel("user cluster")
    plt.ylabel("churned rate")
    plt.show()

    #print(score)
    print("finished")

    exit()