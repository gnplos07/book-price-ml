import pandas as pd

books = pd.read_csv('book.csv')
print(books.head(5))

print(books.shape)
print(books.dtypes)
print(books.isna().sum())