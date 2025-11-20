# Screen Share Script - Pump Anomaly Detection System

## 🎬 Opening (30 seconds)

**[Show: Project root directory]**

"Hey, so this is my pump anomaly detection system. The idea is simple: industrial pumps fail unexpectedly and cost companies thousands of dollars per hour in downtime. I built a system that predicts these failures 10-14 days in advance using machine learning and real-time monitoring."

---

## 📂 Project Structure Tour (1 minute)

**[Show: Project directory tree]**

"Let me show you how it's organized:

- **`data/`** - This is where all the pump sensor data lives. You've got baseline specs in one folder and operational readings in another.

- **`src/`** - This is the core system. I've got modules for data processing, tolerance checking, the ML model, and visualization.

- **`config/`** - Has all the industry-standard tolerance thresholds for different pump types.

- **`models/`** - Where trained ML models get saved.

- **`outputs/`** - Generated reports and visualizations.

- **`diagrams/`** - UML diagrams showing system architecture.

- **`artifacts/`** - Packaged models ready to deploy to edge devices."

---

## 🔧 Core System - Data Processing (1 minute)

**[Open: `src/data_processing.py`]**

"So the first thing the system does is load data. Let me show you `data_processing.py`.

**[Scroll to `load_operational_data` function around line 100]**

This function loads the sensor readings - flow rate, discharge pressure, motor power, pump efficiency. The tricky part is that the real data has spaces in the column names like 'Flow (gpm)' and 'Motor Power (hp)', so I had to handle that exactly.

**[Scroll to `calculate_deviations` function around line 200]**

Then we calculate deviations from baseline. Pretty simple formula:
```
deviation = ((current - baseline) / baseline) * 100
```

So if baseline flow is 500 GPM and current is 575 GPM, that's a +15% deviation."

---

## ⚙️ Tolerance Checking (1 minute)

**[Open: `src/tolerance_checker.py`]**

"Now here's where it gets interesting. Different pump applications have different tolerance thresholds.

**[Scroll to tolerance categories around line 50]**

I implemented 6 different categories from the HI 9.6.3 industry standard:
- Municipal water pumps get ±10% flow tolerance
- API pumps are stricter at ±5%
- Industrial pumps vary based on horsepower

**[Scroll to `check_tolerances` function around line 150]**

This function compares the deviations against the thresholds and determines if the pump is:
- **Normal** - everything within tolerance
- **Warning** - minor violations
- **Critical** - major violations
- **Failure** - severe degradation

The system automatically picks the right category based on pump application and horsepower."

---

## 🤖 Machine Learning Model (1.5 minutes)

**[Open: `src/predictive_model.py`]**

"The coolest part is the predictive model. Let me show you.

**[Scroll to `engineer_features` function around line 100]**

I engineer 64+ features from the raw sensor data:
- Rolling averages over 24 hours and 7 days
- Trend slopes - is flow increasing or decreasing?
- Acceleration - is the degradation speeding up?
- Cross-correlations between parameters

**[Scroll to `train_failure_predictor` function around line 250]**

Then I train a Random Forest model on historical failures. The model learns patterns like:
- When flow increases by 15% and efficiency drops by 3%, failure typically happens in 10-14 days
- Power spikes combined with pressure drops are early warning signs

**[Scroll to `predict_failure` function around line 350]**

The prediction gives us:
- **Remaining Useful Life** in days
- **Failure probability** with confidence scores
- **Contributing factors** - which parameters are most concerning

So instead of waiting for catastrophic failure, maintenance teams get a 2-week heads up."

---

## 🎨 Visualizations (1 minute)

**[Open: `outputs/visualizations/` folder OR show examples]**

"The system generates visualizations automatically.

**[Show multi-parameter dashboard]**

This dashboard shows all 4 key parameters over time - flow, pressure, power, efficiency. The gray bands are the tolerance limits, and you can see exactly when things start going out of bounds.

**[Show degradation timeline]**

This timeline shows when each parameter first exceeded its threshold. You can see efficiency started degrading first, then flow, then pressure - that's the failure pattern.

**[Show generated report]**

And it generates a full markdown report with:
- Timeline of events
- Root cause analysis
- Maintenance recommendations
- ML predictions

All automatic."

---

## 🚀 Edge Deployment - The Cool Part (2 minutes)

**[Open: `src/templates/inference_template.py`]**

"Now here's where Phase 2 comes in. I built this to run on edge devices - basically a $35 Raspberry Pi sitting next to the pump.

**[Scroll to `run_inference` function around line 200]**

The edge device:
1. Reads sensor data from a CSV file every hour
2. Calculates deviations
3. Checks tolerances
4. Runs the ML model prediction
5. If there's an anomaly, it reports to a central API

**[Scroll to `should_report_anomaly` function around line 400]**

It has smart logic to decide when to report:
- If status is Normal, don't report
- If we just reported the same issue 60 minutes ago, don't spam (debouncing)
- If mandatory parameters like flow or pressure are exceeded, definitely report
- If ML predicts failure with high confidence, report

**[Scroll to `format_anomaly_payload` function around line 450]**

When it does report, it sends a detailed JSON payload with:
- Which parameters are violated and by how much
- Current values vs baseline values
- ML prediction and confidence
- Timestamp and pump ID

Let me show you the API client."

---

## 🌐 API Integration (1.5 minutes)

**[Open: `src/anomaly_client.py`]**

"This is the API client that runs on the edge device.

**[Scroll to `submit_anomaly` function around line 100]**

When an anomaly is detected:
1. Formats the JSON payload
2. Sends POST request to Azure API: `/edge/anomalies`
3. Uses bearer token authentication
4. If it fails, it retries with exponential backoff (5s, 10s, 20s delays)
5. If all retries fail, saves locally and tries again next time

**[Scroll to retry logic]**

So the system is resilient - if the internet goes down at a remote pump site, it keeps monitoring and saves anomalies locally. When connectivity comes back, it uploads everything.

**[Open: `config/deployment_config.json`]**

Here's the config file that goes on each edge device:
- API URL and bearer token
- Site ID and pump ID
- Which sensors map to which parameters
- Debounce settings

Everything is configurable per deployment."

---

## 📦 Artifact Packaging (1 minute)

**[Open: `tools/package_artifact.py`]**

"To deploy to edge devices, I built an artifact packager.

**[Scroll to `package` function around line 200]**

It takes:
- The trained ML model (3 MB)
- Baseline parameters for that specific pump
- All the tolerance configurations
- The inference script
- API client
- Requirements file

And packages it into a single ZIP file.

**[Show: `artifacts/` folder if you have a .zip file]**

Then you just:
1. Copy the ZIP to the Raspberry Pi
2. Unzip it
3. Run `python inference.py sensor_data.csv results.json`

And boom, it's monitoring that pump 24/7."

---

## 📊 UML Diagrams (30 seconds)

**[Open: `diagrams/` folder]**

"I also documented the whole system with UML diagrams.

**[Show: System_Class_Diagram.png]**

Class diagram shows all the main components and how they interact.

**[Show: System_Sequence_Diagram.png]**

Sequence diagram shows the flow: edge device reads sensor data, detects anomaly, sends JSON to API.

**[Show: Deployment_Diagram.png]**

And the deployment diagram shows the architecture - edge devices at pump sites, central API in Azure, dashboard for monitoring."

---

## 💻 Quick Demo (1.5 minutes)

**[Open: Terminal or Jupyter notebook]**

"Let me show you it in action. I'll run a quick analysis.

**[Type/show code]:**
```python
from src.pump_monitor import PumpMonitor

# Load pump data
monitor = PumpMonitor('data/raw/baseline/well1_baseline.csv')
monitor.load_operational_data('data/raw/operational/well1_operational.csv')

# Run analysis
monitor.analyze(train_model=True, failure_date='2024-07-25 14:30:00')

# Check status
print(monitor.get_current_status())
# Output: "Critical"

# Get prediction
prediction = monitor.predict_failure()
print(f"RUL: {prediction['remaining_useful_life_days']} days")
print(f"Confidence: {prediction['confidence']}")
# Output: RUL: 12.5 days, Confidence: 0.87

# Generate report
monitor.generate_report('outputs/reports/well1_analysis.md')
```

**[If it runs, show output. If not, show pre-generated outputs]**

So in just a few lines, it:
- Loaded 3 months of sensor data
- Detected that the pump was in critical condition
- Predicted failure 12.5 days in advance with 87% confidence
- Generated a full report with visualizations

**[Open generated report briefly]**

The report tells you:
- When each parameter first exceeded tolerance
- Timeline of degradation
- Which parameters are the leading indicators
- Maintenance recommendations"

---

## 🎯 Wrapping Up (30 seconds)

**[Go back to project root]**

"So to recap:
- **Phase 1** is the core system: data processing, ML training, tolerance checking, visualization
- **Phase 2** is edge deployment: package models, run on Raspberry Pi, send alerts to API
- It predicts failures 10-14 days in advance
- It's resilient - works offline, retries automatically
- It's production-ready - you could deploy this tomorrow

The real value is catching failures before they happen. Instead of a $50,000 emergency repair at 2 AM, you schedule maintenance next Tuesday during planned downtime.

Any questions?"

---

## 🔍 If They Ask Follow-Up Questions

### "How much data do you need?"
"For tolerance detection, just the baseline specs. For ML predictions, at least 7 days of hourly readings, so 168 data points. For training the model, you need historical failures - at least one example."

### "What if the pump is different?"
"The system handles 6 different pump categories automatically. If you have a new type, you just add its tolerance specs to `config/tolerances.json` and it works."

### "How long does it take to run?"
"On a laptop, processing 3 months of data and training the model takes about 30 seconds. On a Raspberry Pi, inference on new data takes under 2 seconds."

### "Can you show the JSON payload?"
**[Open: inference_template.py and scroll to format_anomaly_payload OR open deployment_config.json]**

"Sure, here's what gets sent to the API:
```json
{
  "sourceType": "log",
  "description": "Flow exceeded 15.0% (threshold: 10.0%)",
  "siteId": 35482,
  "pumpId": 1,
  "timestamp": "2024-07-25T14:32:00Z",
  "additionalContext": {
    "status": "Warning",
    "all_deviations": {
      "flow": 15.0,
      "head": 10.2,
      "power": 8.5,
      "efficiency": -1.2
    },
    "baseline_values": {...},
    "current_values": {...}
  },
  "metadata": {
    "modelVersion": "1.0.0",
    "confidence": 0.87,
    "prediction_rul_days": 12.5
  }
}
```
It includes everything the maintenance team needs to diagnose the issue."

### "What happens on the API side?"
"The API receives these anomalies, stores them in a database, and can trigger notifications. In a full deployment, you'd have a dashboard showing all pumps across all sites. I focused on the edge side - the data collection and anomaly detection."

### "Why edge deployment instead of cloud-only?"
"Three reasons:
1. **Latency** - you want to detect anomalies immediately, not wait for cloud processing
2. **Reliability** - remote pump sites have spotty internet, so local processing keeps monitoring working
3. **Cost** - sending hourly sensor data to cloud for thousands of pumps gets expensive fast. Process locally, only send anomalies."

---

## ⏱️ Total Time: ~12 minutes
(Adjust pace based on audience engagement. Skip sections if short on time.)

---

## 📋 Screen Share Checklist

Before you start:
- [ ] Open VS Code with project loaded
- [ ] Have terminal ready (in project directory)
- [ ] Have generated visualizations open in a folder
- [ ] Have diagrams folder ready to show
- [ ] Have example JSON payload ready (copy-paste from code)
- [ ] Close unnecessary windows/tabs
- [ ] Turn off notifications
- [ ] Have backup screenshots if live demo fails

Files to have open in tabs:
1. `README.md`
2. `src/data_processing.py`
3. `src/tolerance_checker.py`
4. `src/predictive_model.py`
5. `src/templates/inference_template.py`
6. `src/anomaly_client.py`
7. `config/deployment_config.json`
8. Terminal/notebook for quick demo

---

**Good luck! Just walk through naturally like you're showing a friend. You built this - you know it best! 🚀**
