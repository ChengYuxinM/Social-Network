import gc
import pandas as pd
from collections import OrderedDict
import json


def userLoad():
    users = {}
    with open("./data/yelp_academic_dataset_user.json", encoding='UTF-8') as lines:
        for u in lines:
            j = json.loads(u)
            if int(j["yelping_since"].split("-")[0]) > 2012:
                users[j["user_id"]] = {"review_count": j['review_count'], "since": j["yelping_since"], "fans": j["fans"],
                                       "friends": j["friends"].split(', '),
                                       "comments": [j["useful"], j["funny"], j["cool"]], \
                                       "compliment": [j["compliment_hot"], j["compliment_profile"],
                                                      j["compliment_note"],
                                                      j["compliment_writer"], j["compliment_photos"]], \
                                       "stars": j["average_stars"], "reviews": [], "friends_score": []}
            #if len(users) >= 100000:
            #    break
            del u, j
    print(len(users))
    return users


def reviewLoad(users):
    # Merge json sets for review and users
    reviews = {}

    with open("./data/yelp_academic_dataset_review.json", encoding='UTF-8') as lines:
        for u in lines:
            j = json.loads(u)
            if j["user_id"] in users:
                reviews[j["review_id"]] = {"funny": j["funny"], "useful": j["useful"], "cool": j["cool"],
                                           "user": j["user_id"], "date": j["date"], "text": j["text"],
                                           "stars": j["stars"]}
                users[j["user_id"]]["reviews"].append(j["review_id"])
            del u, j
    for item in users:
        users[item]["review_count_valid"] = len(users[item]["reviews"])
    print(len(reviews))
    return reviews


def consistentProcess(users, reviews):
    for item in users:
        print(item)
        print(users[item])
        break;

    for item in reviews:
        print(item)
        print(reviews[item])
        break;

    """## remove invalid friend data in user"""

    for item in users:
        name = []
        for friend in users[item]["friends"]:
            if users.get(friend) == None:
                name.append(friend)
        for n in name:
            users[item]["friends"].remove(n)
    return users


def dataPreLoad():
    users = userLoad()
    reviews = reviewLoad(users)
    users = consistentProcess(users, reviews)
    return users, reviews