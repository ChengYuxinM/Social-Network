import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_sentiment_score(senti, review):
    scores = senti.polarity_scores(review)
    return scores


def get_sentiment(score):
    return score['compound']


def sentiment_analysis(reviews):
    senti = SentimentIntensityAnalyzer()

    i = 0
    for item in reviews:
        reviews[item]['sentiment_score'] = get_sentiment_score(senti, reviews[item]['text'])
        reviews[item]['sentiment_vader'] = get_sentiment(reviews[item]['sentiment_score'])
        # print("The reviews score: ",reviews[item]['sentiment_vader'])
        # print("The reviews content: ", reviews[item]['text'])


# def check_results(reviews):
#     for item in reviews:
#         print(reviews[item]['text'])
#         print(reviews[item]['sentiment_score'])
#         print(reviews[item]['sentiment_vader'])
#         break
