import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ndcg_score
import math


def intersection_list(prediction_matrix: pd.DataFrame, true: pd.DataFrame) -> pd.DataFrame:
    true["predicted_rating"] = true.apply(lambda row: prediction_matrix.loc[row['user_id'], row["item_id"]], axis=1)
    return true


def rmse(prediction_matrix: pd.DataFrame, true: pd.DataFrame) -> float:
    ratings = intersection_list(prediction_matrix, true.copy(deep=True))

    return mean_squared_error(ratings['rating'], ratings['predicted_rating'], squared=False)


def precision_at_k(prediction_matrix: pd.DataFrame, test_set: pd.DataFrame, user_mean_ratings: list, k: int):
    """ Calculates precision of the given prediction matrix

    :param prediction_matrix: predicted rating matrix, rows - users | columns - items
    :param test_set: test set ratings, has to have user_id, item_id, rating columns
    :param user_mean_ratings: calculated user mean rating list, length is user number
    :param k: k defines the recommended item list length
    :return: list of corresponding user precisions, average precision
    """

    # filter out all the ratings where the actual value is not known.
    ratings = intersection_list(prediction_matrix, test_set.copy(deep=True))
    precisions = dict()

    # for each user, find their top k predicted ratings
    for user_id in ratings['user_id'].unique():
        user_ratings = ratings[ratings['user_id'] == user_id]
        relevance_threshold = math.floor(user_mean_ratings[user_id - 1])
        recommender_items = user_ratings.nlargest(k, columns='predicted_rating')
        if len(recommender_items) < k:
            pass
        # find number of relevant items at k, here, we filter out the items that are below user mean ratings

        relevant_items = user_ratings[user_ratings['rating'] > min(relevance_threshold, 4)] # not sure if > or >= is more appropriate
        if len(recommender_items) == 0:
            precisions[user_id] = 0
            continue
        precisions[user_id] = len(set(recommender_items["item_id"]).intersection(set(relevant_items["item_id"]))) / len(recommender_items)
    return precisions, sum(precisions.values()) / len(precisions)


def recall_at_k(prediction_matrix: pd.DataFrame, test_set: pd.DataFrame, user_mean_ratings: list, k: int):
    """ Calculates recall of the given prediction matrix

    :param prediction_matrix: predicted rating matrix, rows - users | columns - items
    :param test_set: test set ratings, has to have user_id, item_id, rating columns
    :param user_mean_ratings: calculated user mean rating list, length is user number
    :param k: k defines the recommended item list length
    :return: list of corresponding user recall, average recall
    """

    # filter out all the ratings where the actual value is not known.
    ratings = intersection_list(prediction_matrix, test_set.copy(deep=True))
    recalls = dict()

    # for each user, find their top k predicted ratings
    for user_id in ratings['user_id'].unique():
        user_ratings = ratings[ratings['user_id'] == user_id]
        recommender_items = user_ratings.nlargest(k, columns='predicted_rating')
        if len(recommender_items) < k:
            # print(f"user {user_id} does not have {k} ratings, calculating recall for {len(recommender_items)} ratings")
            pass
        # find number of relevant items at k, here, we filter out the items that are below user mean ratings
        relevance_threshold = math.floor(user_mean_ratings[user_id - 1])
        relevant_items = user_ratings[user_ratings['rating'] > min(relevance_threshold, 4)] # not sure if > or >= is more appropriate
        if len(relevant_items) == 0:
            recalls[user_id] = 0
            continue
        recalls[user_id] = len(set(recommender_items["item_id"]).intersection(set(relevant_items["item_id"]))) / len(relevant_items)
    return recalls, sum(recalls.values()) / len(recalls)


# Make sure that the prediction matrix has users as rows and items as columns.
# Test set should be a DataFrame with a user_id, item_id and rating column.
def calculate_ndcg(prediction_matrix, test_set, k) -> float:

    # Dictionary that will contain the gathered rating pairs.
    paired_ratings = {}

    # Loop over the rows in the test set, saving the true value and retrieving its corresponding prediction
    # from the prediction matrix.
    for index, row in test_set.iterrows():

        user = row[0]
        item = row[1]
        current_true_rating = row[2]

        # Make sure that the indexing for .at is correct.
        # My matrices have strings for row indexing and ints for column indexing.
        # As long as current_predicted_rating gets the value corresponding to the current rating i       # It'll work out.

        current_predicted_rating = prediction_matrix.at[user, item]

        if user not in paired_ratings:

            paired_ratings[user] = ([], [])

        paired_ratings[user][0].append(current_true_rating)
        paired_ratings[user][1].append(current_predicted_rating)

    total_ndcg = 0
    user_count = len(paired_ratings)

    for ratings in paired_ratings.values():

        true_ratings = ratings[0]
        predicted_ratings = ratings[1]

        # If you apply NDCG to a singular input (The user only has 1 rating in the test set) the result is always 1.
        # To reduce the bias this causes, the NDCG score of these inputs is set to 0.5 (the average score).

        if len(true_ratings) == 1:
            total_ndcg += 0.5
            continue

        total_ndcg += ndcg_score([true_ratings], [predicted_ratings], k=k)

    # Return the average NDCG score as answer.
    return total_ndcg / user_count