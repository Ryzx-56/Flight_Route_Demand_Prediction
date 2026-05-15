from FeatureEngineering import feature_engineering
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

df = feature_engineering(
    "International_Report_Departures.csv",
    "International_Report_Passengers.csv"
)

seq_df = df.copy()
seq_df = seq_df.sort_values(["route_id", "Year", "Month"])

high = seq_df["demand"].quantile(0.66)

def label_demand(x):
    if x >= high:
        return "High"
    else:
        return "Low"

seq_df["demand_level"] = seq_df["demand"].apply(label_demand)

label_map = {"Low": 0, "High": 1}
seq_df["demand_level"] = seq_df["demand_level"].map(label_map)

print("data seq head")
print(seq_df[["route_id","Year","Month","demand","demand_level"]].head())

features = [
    "Month",
    "demand"
]
def make_seq(data,seq_len):
    x = []
    y = []

    for i in data["route_id"].unique():
        route = data[data["route_id"]==i]

        values = route[features].values

        for j in range(seq_len,len(values)):
            x.append(values[j-seq_len:j])
            y.append(route["demand_level"].values[j])

    return np.array(x),np.array(y)

x,y=make_seq(seq_df,12)

spliting = int(len(x)*0.8)

Xtrain=x[:spliting]
Xtest=x[spliting:]

Ytrain=y[:spliting]
Ytest=y[spliting:]

Xscaler = MinMaxScaler()

Xtrain_2D = Xtrain.reshape(-1,Xtrain.shape[2])
Xtest_2D = Xtest.reshape(-1,Xtest.shape[2])

Xtrain_scale = Xscaler.fit_transform(Xtrain_2D)
Xtest_scale = Xscaler.transform(Xtest_2D)

Xtrain = Xtrain_scale.reshape(Xtrain.shape[0],Xtrain.shape[1],Xtrain.shape[2])
Xtest = Xtest_scale.reshape(Xtest.shape[0],Xtest.shape[1],Xtest.shape[2])

print(Xtrain.shape)
print(Xtest.shape)
print(Ytrain.shape)
print(Ytest.shape)

model = Sequential()

model.add(GRU(64 , input_shape=(Xtrain.shape[1] , Xtrain.shape[2])))

model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model_train=model.fit(Xtrain , Ytrain , epochs=20 , batch_size=16 , validation_split=0.1)

predictions = model.predict(Xtest)

predictions = (predictions >= 0.5).astype(int).reshape(-1)

accuracy = accuracy_score(Ytest, predictions)

print("actual : ",Ytest[:10])
print("predictions : ",predictions[:10])

print("Accuracy : ",accuracy)
print("Classification Report : ")
print(classification_report(Ytest, predictions, target_names=["Low", "High"]))

print("Confusion Matrix : ")
print(confusion_matrix(Ytest, predictions))

cm = confusion_matrix(Ytest, predictions)

plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["Low", "High"])
plt.yticks([0, 1], ["Low", "High"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.show()

keep_counter = np.sum(predictions == 1)
remove_counter = np.sum(predictions == 0)

print("Depart flights counter : ",keep_counter)
print("Cancel flights counter : ",remove_counter)

labels = ["Depart","Cancel"]
counts = [keep_counter, remove_counter]

total = keep_counter + remove_counter

keep_percent = (keep_counter/total)*100
remove_percent = (remove_counter/total)*100

plt.figure(figsize=(6,6))

bars = plt.bar(labels, counts)

plt.title("Keep VS Remove")
plt.ylabel("Number of Routes")

plt.text(0, keep_counter, f"{keep_percent:.1f}%", ha="center")
plt.text(1, remove_counter, f"{remove_percent:.1f}%", ha="center")

plt.show()
