import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

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
preprocessor = ColumnTransformer(
    transformers=[
        ("condition", condition_encoder, ["condition"]),
        ("number", "passthrough", ["original_price", "years"]),
    ],
    remainder= "drop",
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(train_x,train_y)
predictions = model.predict(test_x)
mae = mean_absolute_error(test_y, predictions)
print(f"Mean Absolute Error: {mae}")

joblib.dump(model, "book_price_model.joblib")
print("Αποθηκεύτηκε το book_price_model.joblib")



