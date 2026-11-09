import pickle 
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#1.
conn=sqlite3.connect("maintenance.db")
df=pd.read_sql("SELECT * FROM telemetry",conn)
conn.close()

#2. features
features=['rpm','operating_hours','temperature_c','vibration_mms']
X=df[features]
y=df['machine_failure']

#3.train/test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#4.model training
model =RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

#5.model evaluation
y_pred=model.predict(X_test)
print("===Model Evaluation Report===")
print(classification_report(y_test,y_pred))

#6. model serialization
with open("model.pkl","wb") as f:
    pickle.dump(model,f)

print('Model serialized and saved to model.pkl')
