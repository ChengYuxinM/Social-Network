import dataLoad
import features
import nlp
import dataProcess
import RNNmodels
import LRmodels

if __name__ == '__main__':
    users, reviews = dataLoad.dataPreLoad()
    print('success data load')
    nlp.sentiment_analysis(reviews)
    print('success semantic analysis')
    dataProcess.labelFeatures2RNN(users,reviews)
    print('success labeled features ')
    features_seq = features.FeaturesProcess2RNN(users,reviews)
    # features.statistic_visualization(features_seq)
    print('success semantic visualization')
    RNNmodels.train(features_seq)