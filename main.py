import pandas as pd

books = pd.read_csv('book.csv')
print(books.head(5))

print(books.shape)
print(books.dtypes)
print(books.isna().sum())

x = books[["title","original_price", "years", "condition"]]
y = books["resale_price"]
