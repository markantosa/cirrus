# CIRRUS

Chemical Intelligence and Residual Heat Recovery for Unified Sustainability

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Integrated machine learning prototype for semiconductor fab sustainability, combining:

- Subsystem A: Wet bath lifecycle prediction and action triggering
- Subsystem B: Waste heat cascade routing and utility matching

This repository contains a fully runnable demo using placeholder synthetic data for poster and presentation use.

## Project Context

- Event: SEMICON Southeast Asia 2026, TECH Zoomers Challenge
- Team: Team 28
- Challenge: Building a Resilient and Sustainable Future for Semiconductor Manufacturing
- Mentors: Cao Wei Zhong, Daniel Soon Chik Chean

## Project Photos

<p align="center">
  <img src="Photos/CIRRUS%20Poster.png" alt="CIRRUS Poster" width="46%" />
  <img src="Photos/Exhibition%20Day.jpeg" alt="CIRRUS Exhibition Day" width="46%" />
</p>

## Problem Framing

Semiconductor fabs face two persistent operational inefficiencies:

- Chemical waste from fixed wet-bath refresh schedules
- Thermal energy waste from high-temperature tool exhaust streams

Most facilities still operate these decisions on static rules and periodic checks. CIRRUS demonstrates a condition-driven alternative with integrated ML inference at the facilities layer.

## System Architecture

### Subsystem A: Smart Wet Bath Lifecycle Extension

- Inputs: pH, ORP, ion concentration, turbidity, temperature, lots processed, bath age
- Model: GradientBoostingRegressor
- Output: Remaining bath life percentage and action trigger
  - CONTINUE
  - TOP-OFF
  - DUMP

### Subsystem B: Waste Heat Cascade Recovery

- Inputs: exhaust temperature, flow rate, DI demand, HVAC demand, ORC capacity, time of day
- Model: RandomForestClassifier
- Output: Routing decision
  - ORC_GENERATION
  - DI_PREHEAT
  - HVAC_SUPPLY
  - BLEND

### Integration Layer

- Shared preprocessing and edge inference workflow
- Unified dashboard output for both subsystems
- Portable execution from one script

## Repository Structure

```text
.
|-- assets/
|-- data/
|   |-- bath_data.csv
|   `-- heat_data.csv
|-- docs/
|   `-- reference/
|-- outputs/
|-- Photos/
|   |-- CIRRUS Poster.png
|   `-- Exhibition Day.jpeg
|-- src/
|   `-- cirrus_model.py
|-- dashboard.html
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Quickstart

### 1) Create and activate environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Run the integrated model

```powershell
python src/cirrus_model.py
```

Data modes:

```powershell
# Auto mode (default): use CSVs if both exist, else synthetic
python src/cirrus_model.py --mode auto

# Force synthetic mode
python src/cirrus_model.py --mode synthetic

# Force real-data mode using defaults in data/
python src/cirrus_model.py --mode real

# Force real-data mode with custom paths
python src/cirrus_model.py --mode real --bath-data data/bath_data.csv --heat-data data/heat_data.csv
```

### 4) Interactive Simulation Dashboard

For live demonstrations and booth environments, use the **CIRRUS Interactive Dashboard**:

- **Location:** `dashboard.html` (Single-file, offline-first)
- **Features:** 
  - Live sliders for Subsystem A & B sensor simulation.
  - Real-time ML inference logic replicated in JavaScript.
  - Sustainability scorecard (Chemicals, Energy, Cost, CO2).
  - **Mobile Access:** Includes a built-in QR code generator to view the dashboard on any mobile device on the same network.

**To run the Interactive Dashboard:**
1. Open `dashboard.html` directly in any modern browser.
2. Alternatively, run the local server for mobile support:
   ```powershell
   # Start the simulation server
   python -m http.server 8080
   ```
3. Open `http://localhost:8080/dashboard.html` and click **"Mobile Link"** to scan the QR code.

If `--mode real` is used, both CSV files are required and validated before training.

### 4) Required CSV schema

Bath dataset (`bath_data.csv`) requires columns:

- `ph`
- `orp_mv`
- `ion_ppm`
- `turbidity`
- `temp_c`
- `lots_run`
- `bath_age_hr`
- `remaining_life_pct`

Optional bath column:

- `action` (auto-derived from `remaining_life_pct` if omitted)

Heat dataset (`heat_data.csv`) requires columns:

- `exhaust_temp_c`
- `flow_rate_m3h`
- `di_demand_kw`
- `hvac_demand_kw`
- `orc_capacity_pct`
- `time_of_day_hr`
- `routing`

Expected outputs:

- Console metrics for Subsystem A and B
- Integrated inference snapshot
- Dashboard image at outputs/cirrus_dashboard.png

## Technical Highlights

- End-to-end ML pipeline for both subsystems
- Synthetic and real-data execution modes
- Schema validation for CSV-based real-data training
- Synthetic dataset generation with domain-informed feature behavior
- Model evaluation built into run flow
- Feature-importance reporting for explainability
- Multi-panel dashboard generation for presentation artifacts

## Additional Documentation

- Sensor hardware list and architecture block diagram: docs/cirrus_sensors_and_block_diagram.md
- Poster-ready top-down diagram PNG: outputs/cirrus_block_diagram_poster.png

## Competition Notes

- Poster format: A1 (841 mm x 594 mm)
- Poster presentation: 10 min presentation + 5 min Q&A
- Judging criteria: strength of idea, commercial viability, teamwork, presentation

## License

This project is licensed under the MIT License. See LICENSE.
