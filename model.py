import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

df = pickle.load(open('df.pkl', 'rb'))


def learn():
    X = df.drop(columns=['Price'])
    y = np.log(df['Price'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

    step1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0, 1, 7, 10, 11])
    ], remainder='passthrough')

    step2 = RandomForestRegressor(n_estimators=100,
                                  random_state=3,
                                  max_samples=0.5,
                                  max_features=0.75,
                                  max_depth=15)

    pipe = Pipeline([
        ('step1', step1),
        ('step2', step2)
    ])

    pipe.fit(X_train, y_train)

    return pipe
