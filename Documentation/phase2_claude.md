# Phase 2B: Anomaly Reporting - Essential Reference

## 🎯 GOAL
Add API reporting to inference.py so edge devices report detected anomalies.

**API:** `https://sp-api-sink.azurewebsites.net/api/v1/edge/anomalies`
**Auth:** `Authorization: Bearer {token}`

---

## 📡 API CALLS

### Submit Anomaly (POST)
```json
{
  "sourceType": "log",
  "description": "Flow exceeded 15% (threshold: 10%)",
  "siteId": 35482,
  "pumpId": 1,
  "sensorId": 101,
  "timestamp": "2024-11-18T14:32:00Z",
  "logValue": 575.0,
  "additionalContext": {
    "status": "Warning",
    "all_deviations": {"flow": 15.0, "head": 3.2, "power": 8.5, "efficiency": -1.2}
  },
  "metadata": {
    "modelVersion": "1.0.0",
    "confidence": 0.87,
    "prediction_rul_days": 12.5
  }
}
```

### Query Anomalies (GET)
```
GET /edge/anomalies?siteId=35482&pumpId=1&page=1&pageSize=25
```

---

## 📋 WHEN TO REPORT

**Report:**
- ✅ Mandatory param (Flow/Head) exceeds tolerance
- ✅ Status = Warning/Critical/Failure
- ✅ ML confidence > 0.7 AND RUL < 7 days

**Don't report:**
- ❌ Normal status
- ❌ Within 1 hour of last report (debounce)

---

## 🔧 NEW FILES

### 1. `src/anomaly_client.py`
```python
class AnomalyAPIClient:
    def __init__(self, base_url, bearer_token):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        })
    
    def submit_anomaly(self, anomaly_data: dict) -> dict:
        """POST with retry logic (3 attempts, exponential backoff)"""
        
    def query_anomalies(self, **filters) -> dict:
        """GET with pagination"""
```

### 2. Update `inference.py` (in artifact)
```python
class EdgeInference:
    def __init__(self):
        # Load API config from deployment_config.json
        self.anomaly_client = AnomalyAPIClient(
            base_url=config["api"]["base_url"],
            bearer_token=config["api"]["bearer_token"]
        )
        self.last_reported = {}  # Debounce tracking
    
    def run_inference(self, input_csv, output_json):
        # Existing code: check tolerances, predict...
        
        # NEW: Report if needed
        if self.should_report_anomaly(results, prediction):
            payload = self.format_anomaly_payload(results, prediction, row)
            try:
                self.anomaly_client.submit_anomaly(payload)
            except Exception as e:
                logger.error(f"API failed: {e}")
                self.save_unsent_anomaly(payload)  # Retry later
```

### 3. Update `config/deployment_config.json`
```json
{
  "api": {
    "base_url": "https://sp-api-sink.azurewebsites.net/api/v1",
    "bearer_token": "YOUR_TOKEN_HERE",
    "retry_attempts": 3
  },
  "site_info": {
    "site_id": 35482,
    "pump_id": 1,
    "sensor_ids": {"flow": 101, "head": 102, "power": 103, "efficiency": 104}
  },
  "anomaly_reporting": {
    "enabled": true,
    "debounce_minutes": 60
  }
}
```

---

## 🔢 STEPS

1. Create `src/anomaly_client.py` with retry logic
2. Update `inference.py` template with reporting
3. Add API config to `deployment_config.json`
4. Test: submit anomaly → query it back
5. Package in artifact

---

## ⚠️ CRITICAL

- **sourceType = "log"** (always)
- **Debounce 60 min** (prevent spam)
- **Include all deviations** (not just violations)
- **Save locally if API down** (retry later)
- **Bearer token = sensitive** (use env var)

---

## ✅ DONE WHEN

- [ ] Can submit to API
- [ ] Can query from API  
- [ ] Inference auto-reports
- [ ] Debouncing works
- [ ] Handles API failures
- [ ] Config includes credentials
- [ ] End-to-end: detect → report → query

---

**Next:** Create `src/anomaly_client.py` with POST/GET methods + retry logic.