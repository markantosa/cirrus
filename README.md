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
|-- docs/
|   `-- reference/
|-- outputs/
|-- src/
|   `-- cirrus_model.py
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

Expected outputs:

- Console metrics for Subsystem A and B
- Integrated inference snapshot
- Dashboard image at outputs/cirrus_dashboard.png

## Technical Highlights

- End-to-end ML pipeline for both subsystems
- Synthetic dataset generation with domain-informed feature behavior
- Model evaluation built into run flow
- Feature-importance reporting for explainability
- Multi-panel dashboard generation for presentation artifacts

## Competition Notes

- Poster format: A1 (841 mm x 594 mm)
- Poster presentation: 10 min presentation + 5 min Q&A
- Judging criteria: strength of idea, commercial viability, teamwork, presentation

## License

This project is licensed under the MIT License. See LICENSE.
