import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from FeatureEngineering import feature_engineering

PATH_DEPARTURES = 'DATA/International_Report_Departures.csv'
PATH_PASSENGERS = 'DATA/International_Report_Passengers.csv'

df = feature_engineering(PATH_DEPARTURES, PATH_PASSENGERS)
df = df.sort_values(['route_id', 'Year', 'Month'])

scaler = MinMaxScaler()
df['scaled_total'] = scaler.fit_transform(df[['Total_Passengers']])

structural_cols = [
    'distance',
    'lag_12',
    'carrier',
    'us_airport',
    'foreign_airport'
    #'avg_demand_per_route'
]

struct_scaler = MinMaxScaler()
df[['distance', 'lag_12']] = struct_scaler.fit_transform(df[['distance', 'lag_12']])

def create_sequences(dataframe, window_size = 12):
    temp_data, struct_data, labels = [], [], []
    for route in dataframe['route_id'].unique():
        route_df = dataframe[dataframe['route_id'] == route]
        if len(route_df) > window_size:
            passengeres = route_df['scaled_total'].values
            struct_features = route_df[structural_cols].values
            for i in range(len(route_df) - window_size):
                temp_data.append(passengeres[i: i + window_size])
                struct_data.append(struct_features[i + window_size])
                label = 1 if passengeres[i + window_size] > 0.2 else 0
                labels.append(label)
    return np.array(temp_data).reshape(-1, window_size, 1), np.array(struct_data), np.array(labels)

X_temp, X_struct, y = create_sequences(df)

X_temp_train, X_temp_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(X_temp, X_struct, y, test_size = 0.2, random_state = 50)

lstm_input = Input(shape = (12, 1), name = 'lstm_input')
x = LSTM(64, return_sequences = False)(lstm_input)
x = Dropout(0.2)(x)
lstm_branch = Dense(32, activation = 'relu')(x)

Dense_input = Input(shape = (5,), name = 'Dense_input')
y = Dense(64, activation = 'relu')(Dense_input)
y = Dropout(0.2)(y)
dense_branch = Dense(32, activation = 'relu')(y)

merged = Concatenate()([lstm_branch, dense_branch])
z = Dense(32, activation = 'relu')(merged)
z = Dropout(0.1)(z)
merged = Dense(16, activation = 'relu')(z)
output = Dense(1, activation = 'sigmoid')(z)

hybrid_model = Model(inputs = [lstm_input, Dense_input], outputs = output)

from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(y_train),
    y = y_train
)

class_weight_dict = {0: weights[0], 1: weights[1]}
print(f"Calculated weights: {class_weight_dict}")

hybrid_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

history = hybrid_model.fit(
    x = [X_temp_train, X_struct_train],
    y = y_train,
    epochs = 50,
    validation_split = 0.2,
    class_weight = class_weight_dict,
    callbacks = [early_stopping],
    verbose = 1
)


y_probs = hybrid_model.predict([X_temp_test, X_struct_test])
y_pred = (y_probs > 0.90).astype(int)

print(classification_report(y_test, y_pred, target_names = ['Remove', 'Keep']))

plt.figure(figsize = (6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = 'd', cmap = 'Blues')
plt.show()



counts = [np.sum(y_pred == 0), np.sum(y_pred == 1)]
labels = ['Remove', 'Keep']

total = sum(counts)
percentages = [(count / total) * 100 for count in counts]

plt.figure(figsize=(8, 5))

bars = plt.bar(labels, counts, color=['steelblue', 'tomato'])

for bar, count, pct in zip(bars, counts, percentages):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + (max(counts) * 0.02),
        f'{count:,}\n({pct:.1f}%)',
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.title('Route Status: Keep vs Remove')
plt.xlabel('Route Status')
plt.ylabel('Number of Routes')

plt.tight_layout()
plt.show()

sample_idx = 97456
history = X_temp_test[sample_idx].flatten()
prediction = y_probs[sample_idx][0]
actual = y_test[sample_idx]

plt.figure(figsize=(10, 5))
plt.plot(range(1, 13), history, marker='o', linestyle='-', color='gray', label='12-Month History')
plt.scatter(13, prediction, color='blue', s=150, marker='*', label=f'Model Probability: {prediction:.2f}')
plt.scatter(13, actual, color='green' if actual == 1 else 'red', s=100, label=f'Actual Target: {actual}')

plt.title(f"Route Spotlight Analysis (Sample #{sample_idx})")
plt.xlabel("Month Sequence")
plt.ylabel("Scaled Passenger Volume")
plt.xticks(range(1, 14), [f'M{i}' for i in range(1, 13)] + ['Target'])
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

