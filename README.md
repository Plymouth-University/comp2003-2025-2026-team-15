[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xGnTrW1S)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=20873272)

# Network Traffic Profiler Dashboard
An interactive dashboard that helps non-experts understand network traffic from PCAP files through automated classification and visualisation.

## Features
- Upload a `.pcap` file for instant analysis
- Automated feature extraction and data cleaning
- Unsupervised ML for detecting abnormal network activity
- Interactive, beginner-friendly dashboard
- Widely supported open-source Python libraries

## Requirements
**Python 3.10+**\
**Virtual Environment (venv)**\

## Installation
Navigate to the /src folder

### 1. Create and activate a virtual environment
`python -m venv venv`\
`venv\Scripts\activate`

### 2. Install dependencies
`pip install -r requirements.txt`

### 3. Start Streamlit local web app
`streamlit run dashboard.py`

## Usage
Examples will be included to demonstrate how to run the dashboard and interpret classification results.

## Tech Stack
PCAP parsing: **scapy**\
Feature extraction: **nfstream**\
CSV data storage: **pandas**\
Dashboard: **streamlit**\
Data visualisation: **plotly**\
Testing: **pytest**\
ML: **scikit-learn**

## Pipeline
PCAP Upload -> Feature Extraction -> Schema Validation -> Dataset Preparation -> ML -> Dashboard Visualisation

## Tests
Basic tests using `pytest` will be included to validate the feature extraction, data processing, and ML pipeline.

## Contributing
This project forms part of a larger project, which aims to develop an AI-driven firewall intrusion detection system, designed for small to medium-sized businesses, and will be further developed in the future.

## Documentation
Find our Design Documents [Here](https://github.com/Plymouth-University/comp2003-2025-2026-team-15/tree/0e4070ed433bc7e53d49b43b76898e0fa8a27ccc/Design%20Documents)

## License
Licensing to be confirmed based on client requirements.