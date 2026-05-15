import pandas as pd 
from Preprocessing import preprocess


def feature_engineering(departures,passengers):

    # Calling the preprocess function from the preprocessing file to get access to the data. 
    df = preprocess(departures,passengers)

    # This code below was written by Nasser.
    import numpy as np

    airports = pd.read_csv("DATA/airports.csv")

    airports = airports.dropna(subset=['iata_code']).drop_duplicates('iata_code').set_index('iata_code')

    df['lat1'] = df['us_airport'].map(airports['latitude_deg'])
    df['lon1'] = df['us_airport'].map(airports['longitude_deg'])
    df['lat2'] = df['foreign_airport'].map(airports['latitude_deg'])
    df['lon2'] = df['foreign_airport'].map(airports['longitude_deg'])

    df["distance"] = ((df["lat1"] - df["lat2"])**2 + (df["lon1"] - df["lon2"])**2)**0.5 * 111

    # 4. Classifying distances by category (1=short, 2=medium, 3=long)
    df["distance_category"] = pd.cut(df["distance"], 
                            bins=[0, 2000, 5000, np.inf], 
                            labels=[1, 2, 3])

    df["distance_category"] = df["distance_category"].cat.codes

    # This code below was written by Abdulmalik 
    avg_demand_per_route = df.groupby(['us_airport', 'foreign_airport'])['Total_Passengers'].transform('mean')
    df['avg_demand_per_route'] = avg_demand_per_route


    # This code was written by Hamed.
    def features_for_SeqData(data):
        grouped = data.groupby(["route_id","Year","Month"])[["Total_Passengers","Total"]].transform("sum")
    
        data["Total_Passengers"] = grouped["Total_Passengers"]
        data["Total"] = grouped["Total"]
    
        data["demand"] = data["Total_Passengers"] / data["Total"]
    
        return data

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error,mean_squared_error

    df["data_dte"]=pd.to_datetime(df["data_dte"])

    df["route"]=df["us_airport"]+"_"+df["foreign_airport"]

    encoder = LabelEncoder()

    df["route_id"]=encoder.fit_transform(df["route"])

    # Encode categorical columns

    le = LabelEncoder()

    categorical_cols = ['carrier', 'us_airport', 'foreign_airport']

    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Fix + encode distance category

    df = features_for_SeqData(df)
    df = df.sort_values(["route_id","Year","Month"])

    df["lag_1"]=df.groupby("route_id")["Total_Passengers"].shift(1)
    df["lag_12"]=df.groupby("route_id")["Total_Passengers"].shift(12)

    df["lag_1"] = df["lag_1"].fillna(0)
    df["lag_12"] = df["lag_12"].fillna(0)

    return df
    