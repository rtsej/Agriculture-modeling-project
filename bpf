import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

X = crops.drop("crop", 1)
y= crops.crop.values

X_train, X_test, y_train, y_test = train_test_split(X, y)

features_dict = {}

for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train[[feature]], y_train)
    
    y_pred = log_reg.predict(X_test[[feature]])
    feature_performance = metrics.f1_score(y_test, y_pred, average="weighted")
    features_dict[feature] = feature_performance
    
best_predictive_feature = {}
best_predictive_feature[max(zip(features_dict.values(), features_dict.keys()))[1]] = features_dict[max(zip(features_dict.values(), features_dict.keys()))[1]]

print(best_predictive_feature)
