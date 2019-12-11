import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# データのインポート
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

test_x = test.copy()

# 特徴量の生成
train_x = train_x.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis=1)
test_x = test_x.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis=1)

# 今回使用するカテゴリ変数にlabel encodingを使用する
for c in ['Sex', 'Embarked']:
    le = LabelEncoder()
    # 欠損値の補完
    le.fit(train_x[c].fillna('NA'))

    # ここでは特徴量を生成している
    # 例えば[apple, grape, orange, apple, orange] => [0, 1, 2, 0, 2]のように変換
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

model = XGBClassifier(n_estimators=20, rnadom_state=71)
model.fit(train_x, train_y)

pred = model.predict_proba(test_x)[:, 1]

pred_label = np.where(pred > 0.5, 1, 0)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index = False)





