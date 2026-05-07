NAMMA-EVCHARGE
AI-Powered EV Load Management & Infrastructure Planning for Bengaluru

NAMMA-EVCHARGE is an AI-driven decision-support system built for BESCOM to optimize electric vehicle (EV) charging patterns and identify strategic locations for new infrastructure. This prototype was developed for the AI for Bharat 2026 initiative to ensure Bengaluru's grid remains resilient during the rapid transition to sustainable mobility.

Project Overview
As EV adoption grows, unmanaged charging creates significant strain on localized grid feeders. NAMMA-EVCHARGE solves this by:

Predicting Demand: Using Machine Learning to forecast 24-hour charging patterns.

Optimizing Schedules: Recommending load-shifting strategies to reduce peak-hour grid stress.

Location Planning: Identifying underserved high-demand hotspots for infrastructure expansion.

Tech Stack
Language: Python 3.13

Framework: Streamlit (Interactive Dashboard)

Machine Learning: Scikit-Learn (Random Forest, DBSCAN)

Data Visualization: Plotly, Matplotlib

Standards: SAE J1772 Charging Profiles

Project Structure
app.py: The frontend dashboard providing BESCOM operators with real-time insights and grid safety alerts.

models.py: The analytical core containing the forecasting and geospatial clustering logic.

data_engine.py: The synthetic data generator grounded in Bengaluru’s zone-specific grid capacities.

Installation & Setup
Follow these steps to run the prototype locally:

Clone the repository:

PowerShell
git clone https://github.com/YourUsername/NAMMA-EVCHARGE.git
cd NAMMA-EVCHARGE


Install dependencies:

PowerShell
python -m pip install streamlit pandas numpy plotly scikit-learn scipy matplotlib
OR
py -m pip install streamlit pandas numpy plotly scikit-learn scipy matplotlib


Run the application:

PowerShell
python -m streamlit run app.py
OR
py -m streamlit run app.py


Non-Negotiables Compliance
Zero Grid Modification: This system acts exclusively as a decision-support layer for planners.

Data Privacy: Operates entirely on synthetic and masked data; no hosted LLMs are used for sensitive information.

Explainable AI: All recommendations are paired with natural language rationales for human operators.

Submission: AI for Bharat 2026 Prototype Challenge
