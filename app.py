import pickle
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance Console", layout="wide"
)
st.title("⚙️ Industrial Equipment Health & Failure Predictor")
st.markdown(
    "Supervised ML pipeline monitoring telemetry signals to prevent machine breakdowns."
)

# Load ML Model
with open("model.pkl", "rb") as f:
  model = pickle.load(f)


# SQLite Query Helper
def run_query(q):
  conn = sqlite3.connect("maintenance.db")
  df = pd.read_sql(q, conn)
  conn.close()
  return df


# --- Top Level Metrics ---
metrics_df = run_query(
    "SELECT COUNT(*) as total_logs, SUM(machine_failure) as total_failures"
    " FROM telemetry"
)
total_logs = metrics_df["total_logs"][0]
total_failures = metrics_df["total_failures"][0]
failure_rate = (total_failures / total_logs) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Telemetry Logs Stored", f"{total_logs}")
c2.metric("Historical Failures Logged", f"{total_failures}")
c3.metric("Fleet Breakdown Rate", f"{failure_rate:.1f}%")

st.divider()

# --- Live Diagnostic Simulator ---
st.header("🔍 Real-Time Sensor Diagnostic & Risk Simulation")
st.write(
    "Adjust sensor readings below to predict the probability of machine"
    " failure in real time:"
)

col_input1, col_input2 = st.columns(2)

with col_input1:
  temp = st.slider(
      "Operating Temperature (°C)",
      min_value=40.0,
      max_value=110.0,
      value=72.0,
  )
  vibration = st.slider(
      "Vibration Level (mm/s)", min_value=1.0, max_value=10.0, value=4.5
  )

with col_input2:
  hours = st.slider(
      "Total Operating Hours", min_value=100.0, max_value=4000.0, value=1200.0
  )
  rpm = st.slider(
      "Rotational Speed (RPM)",
      min_value=1000.0,
      max_value=2200.0,
      value=1500.0,
  )

# Run Prediction
user_data = pd.DataFrame(
    [[rpm, hours, temp, vibration]],
    columns=["rpm", "operating_hours", "temperature_c", "vibration_mms"],
)
failure_probability = model.predict_proba(user_data)[0][1] * 100
is_failure = model.predict(user_data)[0]

st.subheader("Diagnostic Verdict")
if is_failure == 1 or failure_probability > 50:
  st.error(
      f"🚨 **CRITICAL WARNING:** High Breakdown Risk Detected!"
      f" (Failure Probability: {failure_probability:.1f}%)"
  )
elif failure_probability > 25:
  st.warning(
      f"⚠️ **MAINTENANCE ADVISORY:** Elevated wear patterns."
      f" (Failure Probability: {failure_probability:.1f}%)"
  )
else:
  st.success(
      f"✅ **NORMAL OPERATION:** Machine operating within safe parameters."
      f" (Failure Probability: {failure_probability:.1f}%)"
  )

st.divider()

# --- Fleet Analytics Section ---
st.subheader("Historical Fleet Temperature vs. Vibration Distribution")
fleet_df = run_query(
    "SELECT temperature_c, vibration_mms, machine_failure FROM telemetry"
)

fig, ax = plt.subplots(figsize=(8, 4))
scatter = ax.scatter(
    fleet_df["temperature_c"],
    fleet_df["vibration_mms"],
    c=fleet_df["machine_failure"],
    cmap="coolwarm",
    alpha=0.6,
    edgecolors="none",
)
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Vibration (mm/s)")
ax.set_title(
    "Sensor Readings (Blue = Normal, Red = Machine Failure)", fontsize=10
)
st.pyplot(fig)
 