import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

books = pd.read_csv('book.csv', sep= "\t")
print(books.head(5))

print(books.shape)
print(books.dtypes)
print(books.isna().sum())

x = books[["title","original_price", "years", "condition"]]
y = books["resale_price"]

gss  = GroupShuffleSplit(n_splits=1, test_size= 0.2, random_state= 42)
train_idx , test_idx = next(gss.split(x,y, groups= books["title"]))
print(len(train_idx), len(test_idx))

train_x, test_x = x.iloc[train_idx], x.iloc[test_idx]
train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]

condition_encoder = OneHotEncoder(handle_unknown="ignore")




