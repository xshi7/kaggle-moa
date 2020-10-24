import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

labels_dim = 206
lr = 0.01

# read in data
data = pd.read_csv('train_features.csv', header = 0)
labels = pd.read_csv('train_targets_scored.csv', header = 0)
data.loc[:, 'cp_type'] = data.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1}) # ctl_vehicle has no MoA
data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1}) # categorical
data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72:2}) # categorical
data = data.drop(data.columns[0], axis=1).values
labels = labels.drop(labels.columns[0], axis=1).values

# shuffle data and train-test split with ratio 9:1
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42, shuffle = True)

model = Sequential()
model.add(layers.BatchNormalization())
model.add(layers.Dense(2000, activation = 'relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(2000, activation = 'relu'))
model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(labels_dim, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=100, validation_split = 0.1)
model.summary()

pred = model.predict(X_test)
print(pred)
# thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# thresholds = [0.002, 0.01, 0.05, 0.1, 0.4, 0.8]
thresholds = [0.005, 0.01, 0.05, 0.1, 0.3]
for val in thresholds:
    predictions=pred.copy()
    predictions[predictions>=val]=1
    predictions[predictions<val]=0
    # print(predictions)
    correct = 0
    wrong = 0
    f1 = 0 # 2 * (precision * recall) / (precision + recall)
    precision = 0 # tp / (tp + fp)
    recall = 0 # tp / (tp + fn)
    tp_fp = []
    for i in range(len(predictions)):
        tp_fp.append(sum(predictions[i]))
        f1 += metrics.f1_score(y_test[i], predictions[i], average = 'binary')
        precision += metrics.precision_score(y_test[i], predictions[i], average = 'binary')
        recall += metrics.recall_score(y_test[i], predictions[i], average = 'binary')

    f1 /= len(predictions)
    precision /= len(predictions)
    recall /= len(predictions)
    average_pos_predicted = sum(tp_fp) / len(predictions)
    print("Positive predictions by row")
    # print(tp_fp)
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}, Average positive predicted: {:.4f}".format(precision, recall, f1, average_pos_predicted))

