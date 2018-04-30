import pandas as pd
from sklearn.model_selection import train_test_split
# def cleaner1(file1, file2):
#     df = pd.read_csv(file1, sep=" ", header = None)
#     df_reviews = pd.read_csv(file2, sep="delimiter", header = None, engine='python')
#     df['review_text'] = df_reviews[0].astype('str')
#     df['review_length'] = df['review_text'].str.len()
#     df.columns = ['date', 'id_1', 'id_2', "id_3", 'filtered', 'tip1','tip2', 'tip3', 'rating', 'review_text', 'review_length']
#     df[df['filtered'] == -1] = 1
#     df[df['filtered'] == 1] = 0
#     return df
#
# def review_only(file1, file2):
#     df = pd.read_csv(file1, sep=" ", header = None)
#     df_reviews = pd.read_csv(file2, sep="delimiter", header = None, engine='python')
#     df['review_text'] = df_reviews[0].astype('str')
#     df['review_length'] = df['review_text'].str.len()
#     df.columns = ['date', 'id_1', 'id_2', "id_3", 'filtered', 'tip1','tip2', 'tip3', 'rating', 'review_text', 'review_length']
#     df = df[['review_text', 'filtered']]
#     return df

def cleanholdout():
    df = pd.read_csv('Data/YelpNYC/metadata.txt' , sep="\t", header = None)
    df2 = pd.read_csv('Data/YelpNYC/userIdMapping.txt' , sep="\t", header = None)
    df.columns = ['key', 'a', 'b', 'filtered', 'date']
    #double check to get percentage of previously filtered reviews
    df.loc[df['filtered'] == 1, 'filtered'] = 0
    df.loc[df['filtered'] == -1, 'filtered'] = 1
    df2.columns = ['c','key']
    merge1 = pd.merge(df, df2, on='key')
    df3 = pd.read_csv('Data/YelpNYC/reviewContent.txt' , sep="\t", header = None)
    df3.columns = ["key", "blah", "blah2", "review_text"]
    df3['revkey'] = df3['key'].astype(str) + '-' + df3['blah'].astype(str)
    merge1["revkey"] = merge1['key'].astype(str) + '-' + merge1['a'].astype(str)
    df3 = df3.drop(["blah", 'blah2'], axis=1)
    results = pd.merge(merge1, df3, on='revkey')
    results = results.drop(['key_y'], axis=1)
    results.columns = ['userID', 'comp_id', 'rating', 'filtered', 'date', 'user_id', 'user-comp', 'review_text']
    df_prod = pd.read_csv('Data/YelpNYC/productIdMapping.txt' , sep="\t", header = None)
    df_prod.columns = ['comp_name', "comp_id"]
    clean_df = pd.merge(results, df_prod, on="comp_id")
    clean_df['date'] = pd.to_datetime(clean_df['date'])
    clean_df = clean_df.sort_values('date')
    clean_df = clean_df.set_index(clean_df['date'])
    clean_df['review_length'] = clean_df['review_text'].str.len()
    #final = results[['notfraud','review_text']]
    #final.loc[final['recommended'] == -1, 'filtered'] = "Y"
    #final.loc[final['recommended'] == 1, 'filtered'] = "N"
    #final = final.drop('notfraud', axis=1)
    #add review counts
    non_holdout, holdout = train_test_split(clean_df, shuffle=False, test_size=0.1, random_state=0)
    # holdout.to_csv("Data/holdout", sep="\t")
    # non_holdout.to_csv("Data/train_set", sep="\t")
    return non_holdout

def cumulator(clean_df):

    #creates a cumulative review count per business
    clean_df = clean_df.set_index(['comp_id', 'date','userID'], inplace=False, drop=False)
    clean_df['biz_rvws'] =1
    cumsum2 = clean_df.groupby(by=['comp_id', 'date','userID']).sum().groupby(level=[0]).cumsum()
    clean_df = clean_df.drop('biz_rvws', axis=1)
    cumsum2 = pd.DataFrame(cumsum2['biz_rvws'])
    df4 = clean_df.join(cumsum2, how='left', lsuffix='_left', rsuffix='_right')
    #creates a cumulative review count per user, average rating, percentage filtered, average review length

    df5 = df4.set_index(['userID','date', 'comp_id'], inplace=False, drop=False)
    df5['user_rvws'] =1
    cumsum2 = df5.groupby(by=['userID','date', 'comp_id']).sum().groupby(level=[0]).cumsum()
    df5 = df5.drop('user_rvws', axis=1)
    cumsum2['avg_rating'] = cumsum2['rating']/cumsum2['user_rvws']

    cumsum2['past_filt'] = cumsum2['filtered']
    cumsum2.loc[cumsum2['filtered'] > 0, 'past_filt'] -= 1

    cumsum2['percent_filt'] = cumsum2['past_filt']/cumsum2['user_rvws']
    cumsum2['avg_rev_len'] = cumsum2['review_length']/cumsum2['user_rvws']
    cumsum2 = cumsum2.drop(['rating', 'review_length','biz_rvws','filtered'], axis=1)
    df6 = df5.join(cumsum2, how='left', lsuffix='_left', rsuffix='_right')
    return df6
