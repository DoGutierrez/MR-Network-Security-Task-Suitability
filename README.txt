# Reproducibility Materials

This repository contains the anonymized participant data and analysis scripts
for the study "Task–Modality Suitability Across 2D and Mixed Reality Network
Security Interfaces."

## Data

- `Demographics.xlsx` — anonymized participant demographics.
- `Tasks_Performance_Data.xlsx` — task effectiveness, completion time, and
  efficiency data for T1–T4 under the 2D and MR conditions.
- `Task[1-4]_[PC/MR]_Qualitative.xlsx` — post-task questionnaire responses
  and open-ended participant responses for each task and interface condition.

## Analysis

- `statsQuantitative.py` — analyzes task effectiveness, completion time, and
  efficiency for all participants and the security-expert subset.
- `statsSubjective.py` — analyzes the post-task questionnaire responses for
  all participants and the security-expert subset.

Run each script from the directory containing the data files. The scripts
generate the statistical-analysis output files in their respective results
directories.