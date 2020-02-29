import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def naive_models(path):
    print("Loading data file from: {}\n".format(str(path)))
    colnames = ["x", "y"]
    data = pd.read_csv(path, names=colnames, header=None)
    x, y = data['x'].values.reshape(-1, 1), data['y'].values.reshape(-1, 1)

    x_train, x_val, y_train, y_val = train_test_split(x,
                                                      y,
                                                      test_size=0.1,
                                                      random_state=0)

    print("Linear Regression")
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_val, y_pred)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', required=True, help="Data file path")
    args = parser.parse_args()
    naive_models(args.path)