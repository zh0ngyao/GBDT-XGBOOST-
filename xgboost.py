import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging
import xgboost as xgb
import time
from sklearn.datasets import load_iris

logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s', level=logging.INFO)


def XGBoost_LR(df_train):
    X_train = df_train.values[:, :-1]
    y_train = df_train.values[:, -1]
    X_train_xgb, X_train_lr, y_train_xgb, y_train_lr = train_test_split(X_train,
                                                y_train, test_size=0.75)
    XGB = xgb.XGBClassifier(n_estimators = 6）

    XGB.fit(X_train_xgb, y_train_xgb)
    logging.info("训练集特征数据为： \n%s" % X_train_xgb)
    logging.info("训练集标签数据为： \n%s" % y_train_xgb)
    logging.info("转化为叶子节点后的特征为：\n%s" % XGB.apply(X_train_xgb, ntree_limit=0))

    XGB_enc = OneHotEncoder()
    XGB_enc.fit(XGB.apply(X_train_xgb, ntree_limit=0)) # ntree_limit 预测时使用的树的数量
    XGB_LR = LogisticRegression()
    XGB_LR.fit(XGB_enc.transform(XGB.apply(X_train_lr)), y_train_lr.astype('int'))
    X_predict = XGB_LR.predict_proba(XGB_enc.transform(XGB.apply(X_train)))[:, 1]
    AUC_train = metrics.roc_auc_score(y_train.astype('int'), X_predict)
    logging.info("AUC of train data is %f" % AUC_train)


if __name__ == "__main__":
    start = time.clock()
    #加载数据集
    iris=load_iris()
    df_data = pd.concat([pd.DataFrame(iris.data), pd.DataFrame(iris.target)], axis=1)
    df_data.columns = ["x1","x2", "x3","x4", "y"]
    df_new = df_data[df_data["y"]<2]
    logging.info("Train model begine...")
    XGBoost_LR(df_new)
    end = time.clock()
    logging.info("Program over, time cost is %s" % (end-start))