import pandas as pd
from sklearn.model_selection import train_test_split

with open("../Data/ratings.csv",encoding='cp1252') as f:
    df = pd.read_csv(f)

print(df.columns)

print(df['MOS_Complexity'].corr(df['MOS_Understandability']))
print(df['MOS_Complexity'].corr(df['MOS_Lexical_difficulty']))
print(df['MOS_Understandability'].corr(df['MOS_Lexical_difficulty']))



df = df[['ID', 'Sentence', 'MOS_Complexity', 'MOS_Understandability', 'MOS_Lexical_difficulty']]
print(df.columns)

train, test = train_test_split(df, test_size=0.4, shuffle=True)

test, valid = train_test_split(test, test_size=0.5, shuffle=True)

print(train.shape, valid.shape, test.shape)

train.to_csv("../Data/train.csv", encoding='utf-8', index=False)
valid.to_csv("../Data/valid.csv", encoding='utf-8', index=False)
test.to_csv("../Data/test.csv", encoding='utf-8', index=False)