# UML Diagrams - Pump Anomaly Detection System

This directory contains generated UML diagrams for the complete system (Phase 1 + Phase 2B).

## 📊 Diagram Overview

### Class Diagrams (3)

1. **Phase1_Core_Classes.png** - Core analysis components
   - PumpMonitor, DataProcessor, ToleranceChecker
   - PredictiveModel, FeatureEngineer, Visualizer
   - Data models (BaselineData, OperationalReading, DeviationRecord)

2. **Phase2B_Edge_Classes.png** - Edge deployment components
   - EdgeInference, AnomalyAPIClient, DebounceManager
   - ArtifactPackager, AnomalyQueryTool, LocalStorageManager

3. **Complete_System_Classes.png** - Full system overview
   - All 25+ classes
   - Relationships between Phase 1 and Phase 2B
   - Configuration management

### Use Case Diagrams (3)

4. **Phase1_UseCases.png** - Core analysis use cases (11)
   - Load data, calculate deviations, check tolerances
   - Train ML model, predict failure
   - Generate visualizations and reports

5. **Phase2B_UseCases.png** - Edge deployment use cases (13)
   - Package artifact, deploy to edge
   - Run inference, detect anomalies
   - Report to API, handle failures
   - Query and visualize anomalies

6. **Complete_System_UseCases.png** - Complete workflow
   - All actors (Analyst, DevOps, Edge Device, Dashboard)
   - End-to-end workflows

### Sequence Diagrams (4)

7. **Phase1_Training_Sequence.png** (281 KB) - Training and analysis workflow
   - Complete flow from data loading to report generation
   - Feature engineering (64 features)
   - Model training with Random Forest
   - Visualization generation
   - 80+ interaction steps

8. **Phase2B_Inference_Sequence.png** (43 KB) - Edge inference workflow
   - System initialization
   - Hourly inference execution
   - Anomaly detection logic
   - API submission with retry (3 attempts, exponential backoff)
   - Debouncing mechanism
   - Graceful degradation
   - 100+ interaction steps

9. **Artifact_Packaging_Sequence.png** (314 KB) - Artifact creation
   - Baseline loading
   - Model file copying
   - Configuration file creation
   - ZIP archive generation

10. **Query_Anomalies_Sequence.png** (353 KB) - Anomaly query workflow
    - Paginated API queries
    - DataFrame parsing
    - Summary statistics
    - CSV export
    - Timeline visualization

### System Diagrams (2)

11. **Component_Diagram.png** - System components
    - 9 major components
    - Data flow between modules
    - Phase 1 ↔ Phase 2B integration

12. **Deployment_Diagram.png** - Infrastructure
    - Central analysis server
    - Edge devices (Raspberry Pi)
    - Azure cloud (API + Database)
    - Network topology

## 🎨 Viewing the Diagrams

### In This Directory
All diagrams are PNG images and can be viewed directly:

```bash
# Open in default image viewer (macOS)
open Phase1_Core_Classes.png

# Or view all diagrams
open *.png
```

### In Documentation
See `UML_DIAGRAMS.md` in the project root for:
- PlantUML source code
- Detailed explanations
- Implementation notes

## 📐 Re-generating Diagrams

If you modify `UML_DIAGRAMS.md`, regenerate diagrams with:

```bash
cd ..
python3 generate_diagrams.py
```

**Requirements:**
- Java (for PlantUML)
- `plantuml.jar` in project root
- Python 3.8+

## 📊 Diagram Statistics

| Diagram Type | Count | Total Size |
|--------------|-------|------------|
| Class        | 3     | ~28 KB     |
| Use Case     | 3     | ~27 KB     |
| Sequence     | 4     | ~990 KB    |
| System       | 2     | ~19 KB     |
| **Total**    | **12**| **~1.1 MB**|

## 🔍 Diagram Details

### Class Diagrams
- **Total Classes**: 25+
- **Key Patterns**: Factory, Strategy, Observer
- **Relationships**: Inheritance, Composition, Aggregation

### Use Case Diagrams
- **Total Use Cases**: 24
- **Actors**: 5 (Analyst, DevOps, Edge Device, Dashboard, API)
- **Relationships**: Include, Extend

### Sequence Diagrams
- **Total Interactions**: 300+
- **Participants**: 15+
- **Depth**: Up to 10 levels of nesting

## 📝 Notes

- All diagrams generated from PlantUML source
- High resolution (300 DPI)
- Optimized for documentation and presentations
- Source diagrams in `UML_DIAGRAMS.md`

---

**Generated**: November 19, 2025
**Tool**: PlantUML v1.2024.3
**Format**: PNG
