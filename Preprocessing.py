import pandas as pd 

def preprocess(path1,path2):
      # Load the data
      departures = pd.read_csv(path1)
      passengers = pd.read_csv(path2)


      #Merge Two files together for better machine learning model accuracy
      df = pd.merge(
      departures,
      passengers,
      on =["Year","Month","usg_apt","fg_apt","carrier"],
      how = 'left'
      )

      # Make the data in order by year, month 
      df = df.sort_values(by=['Year', 'Month'])

      # we have lots of new columns that were added from merge so we have to fix that 

      #Remove _x from the end of the column names to increase readability
      #_x was added from the merge.
      df.columns = [col[:-2] if col.endswith('_x') else col for col in df.columns]

      # Rename the columns we will use to make them more readable and easier to understand
      df = df.rename(columns={
      'Scheduled_y': 'Scheduled_Passengers',
      'Charter_y': 'Charter_Passengers',
      'Total_y': 'Total_Passengers',
      'Scheduled_x': 'Scheduled_Departures',
      'Charter_x': 'Charter_Departures',
      'Total_x': 'Total_Departures',
      'usg_apt_id': 'us_airport_id',
      'usg_apt': 'us_airport',
      'usg_wac': 'us_world_area_code',
      'fg_apt_id':'foreign_airport_id',
      'fg_apt': 'foreign_airport',
      'fg_wac': 'foreign_world_area_code'
      })

      # Drop the columns we don't need (Reapeated columns from the merge. thats why they all end in _y)
      cols_to_drop = [
      "data_dte_y",
      "usg_apt_id_y",
      "usg_wac_y",
      "fg_apt_id_y",
      "fg_wac_y",
      "airlineid_y",
      "carriergroup_y",
      "type_y"
      ]

      df = df.drop(columns=cols_to_drop)

      # Drop the type column because it says departure for every single row
      df = df.drop(columns=["type"])


      # we have 3055 missing values in carrier 
      # 249835 missing values in Scheduled_passengers
      # 249835 missing values in Charter_passengers
      # 249835 missing values in total_passengers

      # 3055 missing values in carrier is low compared to data size of 930k rows so we just gonna drop it 
      df = df.dropna(subset=['carrier'])

            # the values 249835 are missing on all 3 Scheduled_passengers,Charter_passengers, total_passengers
            # So we check that all 3 are missing on the same rows ? if yes makes it much harder to predict the values 
            #print(df[["Scheduled_Passengers","Charter_Passengers","Total_Passengers"]].isna().all(axis=1).sum())
            # After we run this code we got 249577 back so that means there is 258 rows that arent missing all 3 values 
            # We will fill those 258 rows


      df["Scheduled_Passengers"] = df["Scheduled_Passengers"].fillna(df["Total_Passengers"] - df["Charter_Passengers"])
      df["Charter_Passengers"] = df["Charter_Passengers"].fillna(df["Total_Passengers"] - df["Scheduled_Passengers"])
      df["Total_Passengers"] = df["Total_Passengers"].fillna(df["Scheduled_Passengers"] + df["Charter_Passengers"])


      # group the data together based on departure airport and destination airport, airline ID and the month to guess some of the missing data 
      # we will be using the median instead of the mean just in case the data is skewed.

      group_cols = ['us_airport_id', 'foreign_airport_id', 'airlineid', 'Month']

      for col in ['Scheduled_Passengers', 'Charter_Passengers', 'Total_Passengers']:
            medians = df.groupby(group_cols)[col].transform('median')
            df[col] = df[col].fillna(medians)

           

      # we have gone down from 249835 missing values to 233004 . using a wider group for the missing values not fixed by groupby
      group_cols_broad = ['us_airport_id', 'foreign_airport_id', 'Month']

      for col in ['Scheduled_Passengers', 'Charter_Passengers', 'Total_Passengers']:
            medians = df.groupby(group_cols_broad)[col].transform('median')
            df[col] = df[col].fillna(medians)


      # after the broder group i went down from missing 233004 to only missing 76925. 
      # going to use an even broader group to fill them in 
      group_cols_broader = ['us_airport_id', 'foreign_airport_id']

      for col in ['Scheduled_Passengers', 'Charter_Passengers', 'Total_Passengers']:
            medians = df.groupby(group_cols_broader)[col].transform('median')
            df[col] = df[col].fillna(medians)


      # after the broader group we went down from missing 76925 to missing 42948
      #drop the remaining 42948 values because we did many groups and they must be rare routes or something. 
      # we managed to recover about 207K rows so losing 42k shouldnt be a big problem
      df = df.dropna(subset=['Scheduled_Passengers','Charter_Passengers','Total_Passengers'])

      return df
