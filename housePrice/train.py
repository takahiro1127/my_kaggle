import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

train = pd.read_csv('/data/train.csv')
test = pd.read_csv('/data/test.csv')

train.is_null().sum()[train.is_null().sum() > 0].sort_values(ascending = False)

alldata[na_col_list].dtypes.sort_values()  # データ型
na_float_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes == 'float64'].index.tolist()  # float64
na_obj_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes == 'object'].index.tolist()  # object

for na_float_col in na_float_cols:
    alldata.loc[alldata[na_float_col].isnull(), na_float_col] = 0.0
for na_obj_col in na_obj_cols:
    alldata.loc[alldata[na_obj_col].isnull(), na_obj_col] = 'NA'

# カテゴリカル変数の特徴量をリスト化
cat_cols = alldata.dtypes[alldata.dtypes == 'object'].index.tolist()
# 数値変数の特徴量をリスト化
num_cols = alldata.dtypes[alldata.dtypes != 'object'].index.tolist()
# データ分割および提出時に必要なカラムをリスト化
other_cols = ['Id', 'WhatIsData']
# 余計な要素をリストから削除
cat_cols.remove('WhatIsData')  # 学習データ・テストデータ区別フラグ除去
num_cols.remove('Id')  # Id削除
# カテゴリカル変数をダミー化
alldata_cat = pd.get_dummies(alldata[cat_cols])
# データ統合
all_data = pd.concat(
    [alldata[other_cols], alldata[num_cols], alldata_cat], axis=1)


scaler = StandardScaler()  # スケーリング
param_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # パラメータグリッド
cnt = 0
for alpha in param_grid:
    ls = Lasso(alpha=alpha)  # Lasso回帰モデル
    pipeline = make_pipeline(scaler, ls)  # パイプライン生成
    X_train, X_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=0.3, random_state=0)
    pipeline.fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(
        y_train, pipeline.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))
    if cnt == 0:
        best_score = test_rmse
        best_estimator = pipeline
        best_param = alpha
    elif best_score > test_rmse:
        best_score = test_rmse
        best_estimator = pipeline
        best_param = alpha
    else:
        pass
    cnt = cnt + 1

print('alpha : ' + str(best_param))
print('test score is : ' + str(best_score))
