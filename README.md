# LABRADOR — Lab Value Analysis for Rare Diseases

**Laboratory for Rare Disease and Observation Studies**

*A collaboration between LABRADOR, the [Health Research Data Portal (FDPG)](https://forschen-fuer-gesundheit.de), and the [PrivateAIM/FLAME](https://privateaim.de) consortium.*

---

## Background

In Germany, approximately four million people live with a rare disease. Despite this high overall prevalence, the average time from first symptoms to a confirmed diagnosis spans several years — four years for children, eight years for adults. During this time, patients routinely receive standard lab work that often contains characteristic, disease-indicative abnormalities — which go unrecognized due to lack of awareness or the absence of digital decision support.

**LABRADOR investigates whether routinely collected lab parameters carry a clinically relevant Positive Predictive Value (PPV) for specific rare diseases**, enabling them to serve as digital early-warning signals in the diagnostic process.

---

## Target Diseases

| Disease | ORPHA | Key Lab Marker(s) | Goal |
|---|---|---|---|
| Familial Hypophosphatasia (HPP) | ORPHA:436 | Persistently low ALP | Diagnosis support |
| Homozygous Familial Hypercholesterolemia (hoFH) | ORPHA:391665 | Strongly elevated LDL | Diagnosis support |
| Myotonic Dystrophy Type 1 (DM1) | ORPHA:273 | Elevated CK, AST, ALT, HbA1c; low IgG, TSH | Diagnosis support |
| Spinal Muscular Atrophy (SMA) | ORPHA:83330–83420 | ALT/AST, platelet count | Therapy safety monitoring |

> **Note on SMA**: Unlike the other three diseases where the focus is on reducing diagnostic delay, the SMA analysis targets safety monitoring of already-diagnosed patients receiving gene therapy (Onasemnogen-Abeparvovec / Zolgensma®), where liver enzymes and platelet counts serve as markers for adverse drug effects.

---

## Data Infrastructure

Analyses draw on routine care data from the **Data Integration Centers (DIZ)** of German university hospitals, accessed via the **Health Research Data Portal (FDPG)**. Data is standardized to the **HL7 FHIR** standard and lab values are identified by **LOINC codes**.

MII Core Data Set modules used:
- `DIAGNOSE` — ICD-10-GM and Orpha codes
- `LABORBEFUND` — Lab values with timestamps
- `PERSON` — Age, sex, region

---

## Analysis Plan

Each target disease is analyzed in six steps:

| Step | Description |
|---|---|
| 1 | **Descriptive Epidemiology** — Case counts by age, sex, and DIZ site |
| 2 | **Distribution Analysis** — Density plots of disease-specific lab parameter in general population vs. disease group |
| 3 | **Threshold Determination** — Optimal lab cutoff via Youden Index (maximizing Sensitivity + Specificity − 1) |
| 4 | **Diagnostic Quality Criteria** — Prevalence, Sensitivity, Specificity, PPV for the single lab parameter |
| 5 | **Multiparametric Test** — Whether combining multiple lab parameters increases PPV (adjusted for age, sex, confounders) |
| 6 | **Deductive Case Finding** — Applying the derived lab signature to the remaining population to flag potentially undiagnosed patients |

### Comparison Groups

For each target disease, analyses are run against two populations:
- **Reference group 1**: all patients in the DIZ data without that diagnosis
- **Reference group 2 (cluster)**: patients with clinically or lab-chemically similar differential diagnoses

Steps 3–6 are primarily conducted against Reference group 2, as a clinically meaningful PPV is only interpretable in comparison to a phenotypically similar population. The cluster approach reduces false-positive signals that inevitably arise when comparing against the entire remaining population.

---

## Two-Lane Architecture

LABRADOR operates in two complementary analysis modes:

```
┌─────────────────────────────────────────────────────────────────┐
│  Lane 1 · Single-Center (Direct)                                │
│                                                                 │
│  Run locally at one DIZ                                         │
│  Script: densities_groupedLabValues.py                          │
│  Output: density plots + group summary CSVs → ZIP               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Lane 2 · Multi-Center (Federated via FLAME)                    │
│                                                                 │
│  Distributed across multiple DIZ nodes — no raw data transfer  │
│  Script: flame_analyzeSyntheticLabValues.py                     │
│  Platform: PrivateAIM / FLAME (StarModel pattern)               │
│  Output: aggregated density plots + CSVs → ZIP                  │
└─────────────────────────────────────────────────────────────────┘
```

### Lane 1 — Single-Center Direct Analysis

[`densities_groupedLabValues.py`](densities_groupedLabValues.py) runs directly at a single DIZ. It:

1. Filters lab values by LOINC code
2. Keeps only positive values (QA filter)
3. Sorts values and groups them into **consecutive, non-overlapping chunks of 10**
4. Drops any incomplete final chunk
5. Computes the **mean per chunk**
6. Generates a **weighted KDE density plot** of the chunk means
7. Exports group summaries and plots into a ZIP file

The chunk-of-10 grouping is a privacy-aware aggregation that avoids exposing individual patient values in the output.

**Currently implemented lab values:** ALP (HPP), LDL (hoFH)

### Lane 2 — Federated Analysis via FLAME

[`flame_analyzeSyntheticLabValues.py`](flame_analyzeSyntheticLabValues.py) uses the **FLAME platform** from the [PrivateAIM consortium](https://privateaim.de), which enables federated analyses across multiple DIZ nodes without transferring patient-level data. Each DIZ runs the analysis locally; only aggregated results are sent to a central node.

The script follows the **StarModel** pattern:

- **`LabDataAnalyzer`** *(runs at each DIZ node)*: filters lab values by LOINC code; returns filtered ALP and LDL dataframes as node result
- **`LabDataAggregator`** *(runs at the central aggregator node)*: receives filtered dataframes from all nodes, concatenates them, applies the chunk-of-10 grouping, and generates density plots and CSVs into a ZIP

**Currently implemented lab values:** ALP (HPP), LDL (hoFH)

---

## Repository Structure

```
LabValues/
├── densities_groupedLabValues.py           # Lane 1: single-center analysis
├── flame_analyzeSyntheticLabValues.py      # Lane 2: FLAME federated analysis
├── analyzeSyntheticLabValues.py            # Utility: run analysis on synthetic dataset
├── synth_dataset.csv                       # Synthetic test dataset
├── lab_density_outputs.zip                 # Example output
└── docs/
    └── roadmap.md                          # Planned next steps
```

---

## Current Status

- [x] Lane 1 — implemented and validated on synthetic data (ALP, LDL)
- [x] Lane 2 — FLAME script implemented and validated on synthetic data (ALP, LDL)
- [ ] Real DIZ data access via FDPG (application pending)
- [ ] Steps 3–6: threshold determination, PPV, multiparametric test, deductive case finding
- [ ] Extension to DM1 lab markers (CK, AST, ALT, HbA1c, IgG, TSH)
- [ ] Extension to SMA safety monitoring markers (ALT/AST, platelets)

See [`docs/roadmap.md`](docs/roadmap.md) for the detailed plan.

---

## Dependencies

```bash
pip install pandas numpy seaborn matplotlib
```

For Lane 2 (FLAME), the PrivateAIM FLAME SDK is required — see the [PrivateAIM documentation](https://privateaim.de) for installation.

---

## Related

- [FDPG — Health Research Data Portal](https://forschen-fuer-gesundheit.de/)
- [PrivateAIM / FLAME Platform](https://privateaim.de)
- [Medizininformatik-Initiative (MII)](https://www.medizininformatik-initiative.de)
- [NAMSE — National Action Alliance for People with Rare Diseases](https://www.namse.de)
