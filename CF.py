import pandas as pd
import numpy as np

def predict_CF(df_ratings:pd.DataFrame, df_simil:pd.DataFrame, n:int, user:int, item:int):
    try:
        if np.isnan(df_ratings[item][user]):
            # select all not nan column indices for the selected item row
            selected_rows = df_ratings[item][~df_ratings[item].isnull()].index
            # select similarity scores for user and users who rated the item
            simil = df_simil.loc[user][selected_rows].nlargest(n)
            simil = simil.dropna()
            # if there are no similar users who did the item return the user mean
            if len(simil) == 0:
                return df_ratings.loc[user].mean()
            denom = 0
            nom = 0
            for ind in simil.index:
                nom += simil.loc[ind] * df_ratings[item][ind]
                denom += np.abs(simil.loc[ind])
            return nom/denom

        else:
            # the user already rated an item so return this score
            return df_ratings[item][user]
    except KeyError:
        return df_ratings.loc[user].mean()
