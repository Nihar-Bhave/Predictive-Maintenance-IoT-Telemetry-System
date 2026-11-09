import numpy as np
import pandas as pd

np.random.seed(42)
n_records = 1000

#1.
machine_ids = np.random.choice([101,102,103,104,105],size=n_records)
rpm= np.random.normal(loc=1500,scale=120,size=n_records).round(1)
operating_hours=np.random.uniform(100,3500,size=n_records).round(1)
temperature_c = np.random.normal(loc=68,scale=10,size=n_records).round(1)
vibration_mms = np.random.normal(loc=4.5,scale=1.2,size=n_records).round(2)

#2.
failure_risk_score=(
   (temperature_c > 82).astype(int)*3 
   + (vibration_mms > 6.2).astype(int)*4 
   + (operating_hours > 2800).astype(int)*2 
   +  np.random.binomial(1,0.05,size=n_records)
   )
machine_failure = np.where(failure_risk_score >= 4,1,0)


#3.
df_telemetry = pd.DataFrame({
'log_id': range(1,n_records+1),
'machine_id':machine_ids,
'rpm':rpm,
'operating_hours':operating_hours,
'temperature_c':temperature_c,
'vibration_mms':vibration_mms,
'failure_risk_score':failure_risk_score,
'machine_failure':machine_failure
})
df_telemetry.to_csv('telemetry_data.csv',index=False)
print(
    f"Generated 1,000 sensor logs with {machine_failure.sum()} simulated failure events."
)
