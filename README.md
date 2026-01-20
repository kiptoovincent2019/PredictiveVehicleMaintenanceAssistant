# Predictive Vehicle Maintenance Assistant

**Predictive Vehicle Maintenance Assistant** is a Python Tkinter-based application that allows automotive professionals, fleet managers, and vehicle owners to run advanced diagnostics using OBD-II fault codes. It predicts likely active faults, provides safety guidance, and recommends maintenance actions—all with multi-language support (English, Spanish, French).

---

## Features

- Run diagnostics on single vehicles or multiple vehicles (Fleet Mode)
- Supports all known standard and manufacturer-specific OBD-II fault codes
- Provides actionable guidance for unknown codes:
  - Consult manufacturer service manuals
  - Use recommended diagnostic tools
  - Standard troubleshooting steps and safety advice
- Probability-based fault scoring and predicted service dates
- Multi-language support: English, Spanish, French
- Export diagnostic reports to PDF or plain text
- Historical logging of past diagnostics
- User-friendly GUI with safety alerts panel and fault summaries

---

## Installation

1. pip install matplotlib reportlab pandas


2. Clone the repository:

```bash
git clone https://github.com/kiptoovincent2019/PredictiveVehicleMaintenanceAssistant.git
cd PredictiveVehicleMaintenanceAssistant

3. Create and activate a virtual environment:
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

4. Install dependencies:
pip install -r requirements.txt

Usage

Run the GUI:

python PredictiveVehicleMaintenanceAssistant.py
