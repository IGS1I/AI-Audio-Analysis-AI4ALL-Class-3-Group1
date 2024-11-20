import pandas as pd

df1 = pd.read_csv('EmoMusic_features_45_sec.csv')
df2 = pd.read_csv('GTZAN_features_3_sec.csv')
df3 = pd.read_csv('IRMAS_features_2_sec.csv')

combined_df = pd.concat([df1, df2, df3], ignore_index=True)

combined_df['label'] = ['happy', 'sad', 'jazz', 'rock'] * (len(combined_df) // 4)

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(combined_df, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(train_data.iloc[:, :-1])  # Exclude the label column
X_val = scaler.transform(val_data.iloc[:, :-1])
X_test = scaler.transform(test_data.iloc[:, :-1])
