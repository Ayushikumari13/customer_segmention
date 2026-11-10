Customer Segmentation Project
=============================
report/  - Final report (PDF and Word) with method, results, charts, code appendix
code/    - generate_data.py (creates sample data), segmentation_analysis.py (K-Means analysis)
data/    - customers_raw.csv, customers_segmented.csv, segment_profile.csv
charts/  - PNG charts used in the report

To reproduce: put code and data files in one folder, then run
  python generate_data.py
  python segmentation_analysis.py
Requires: pandas, scikit-learn, matplotlib, seaborn
