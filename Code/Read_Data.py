import pandas as pd
from sklearn.model_selection import train_test_split

with open("../Data/ratings.csv",encoding='cp1252') as f:
    df = pd.read_csv(f)

print(df.columns)

print(f"The correlation coefficient between Complexity of Understandability is {df['MOS_Complexity'].corr(df['MOS_Understandability'])}")
print(f"The correlation coefficient between Complexity of Lexical_difficulty is {df['MOS_Complexity'].corr(df['MOS_Lexical_difficulty'])}")
print(f"The correlation coefficient between Understandability of Lexical_difficulty is {df['MOS_Understandability'].corr(df['MOS_Lexical_difficulty'])}")



df = df[['ID', 'Sentence', 'MOS_Complexity', 'MOS_Understandability', 'MOS_Lexical_difficulty']]
print(df.columns)

train, test = train_test_split(df, test_size=0.4, shuffle=True)
test, valid = train_test_split(test, test_size=0.5, shuffle=True)

print(f"There are {train.shape[0]}, {valid.shape[0]}, and {test.shape[0]} instances in the train, validation and test sets, respectively.")

train.to_csv("../Data/train.csv", encoding='utf-8', index=False)
valid.to_csv("../Data/valid.csv", encoding='utf-8', index=False)
test.to_csv("../Data/test.csv", encoding='utf-8', index=False)
print(f"Saved in the files!")