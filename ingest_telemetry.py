import sqlite3
import pandas as pd

#i.
df=pd.read_csv('telemetry_data.csv')

#ii.
conn =sqlite3.connect('maintenance.db')
#iii.
df.to_sql("telemetry",conn,if_exists='replace',index=False)


cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) ,SUM(machine_failure) FROM telemetry")
total_logs, failures = cursor.fetchone()
conn.close()

print(f"Database 'maintenance.db' initialized: {total_logs} logs , {failures} failures logged.")

