import pandas as pd
from sklearn.model_selection import train_test_split

def cleanholdout():
    #Mapping data from 3 files together and labeling columns
    df = pd.read_csv('Data/YelpNYC/metadata.txt' , sep="\t", header = None)
    df2 = pd.read_csv('Data/YelpNYC/userIdMapping.txt' , sep="\t", header = None)
    df.columns = ['key', 'a', 'b', 'filtered', 'date']
    df.loc[df['filtered'] == 1, 'filtered'] = 0
    df.loc[df['filtered'] == -1, 'filtered'] = 1
    df2.columns = ['c','key']
    merge1 = pd.merge(df, df2, on='key')
    df3 = pd.read_csv('Data/YelpNYC/reviewContent.txt' , sep="\t", header = None)
    df3.columns = ["key", "dropping", "dropping2", "review_text"]
    df3['revkey'] = df3['key'].astype(str) + '-' + df3['dropping'].astype(str)
    merge1["revkey"] = merge1['key'].astype(str) + '-' + merge1['a'].astype(str)
    df3 = df3.drop(["dropping", 'dropping2'], axis=1)
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
    clean_df.drop('date.1', 'comp_name', 'user_id', 'user-comp', 'comp_name', axis=1)
    #create holdout and training files
    non_holdout, holdout = train_test_split(clean_df, shuffle=False, test_size=0.1, random_state=0)
    holdout.to_csv("Data/holdout.csv", sep="\t")
    non_holdout.to_csv("Data/train_set.csv", sep="\t")

if __name__ == "__main__":
    cleanholdout()
