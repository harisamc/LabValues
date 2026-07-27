# Roadmap

This document describes the planned development of LABRADOR beyond the current proof-of-concept scripts.

---

## Where We Are Now

The current scripts implement **Step 2** of the six-step analysis plan: distribution analysis via density plots.

Both lanes are operational on synthetic data for **ALP** (Hypophosphatasia) and **LDL** (homozygous Familial Hypercholesterolemia):

| Script | Status |
|---|---|
| `densities_groupedLabValues.py` (Lane 1, single-center) | Done — validated on synthetic data |
| `flame_analyzeSyntheticLabValues.py` (Lane 2, FLAME federated) | Done — validated on synthetic data |

---

## Analysis Steps Still to Implement

### Step 3 — Threshold Determination
Derive an optimal lab cutoff value per disease using the **Youden Index**:

```
J = Sensitivity + Specificity − 1  →  maximized
```

This will be applied both against the full reference population and against the cluster-based differential diagnosis population.

### Step 4 — Diagnostic Quality Criteria
For each target disease and lab parameter, compute:
- Prevalence in the DIZ population
- Sensitivity and Specificity at the Youden-optimal threshold
- **Positive Predictive Value (PPV)**

### Step 5 — Multiparametric Test
Test whether combining multiple lab parameters improves the PPV, adjusted for age, sex, and relevant confounders. Relevant for DM1, where the disease signature involves multiple markers.

### Step 6 — Deductive Case Finding
Apply the derived lab signature patterns to the remaining (undiagnosed) DIZ population to identify patients who may have an unrecognized rare disease.

---

## Disease Expansion

### Currently implemented
- **HPP** — ALP (LOINC: `109532-2`, `1783-0`, `59164-4`, `16337-8`)
- **hoFH** — LDL (LOINC: `13457-7`, `53133-5`, `96258-9`, `69419-0`)

### Planned
- **DM1 (Myotonic Dystrophy Type 1, ORPHA:273)**
  - Markers: CK, AST, ALT, HbA1c (elevated); IgG, TSH (decreased)
  - Multi-marker signature → relevant for Step 5

- **SMA (Spinal Muscular Atrophy, ORPHA:83330–83420)**
  - Markers: ALT, AST, platelet count
  - Goal: therapy safety monitoring for patients on Onasemnogen-Abeparvovec (Zolgensma®), not diagnosis
  - Cluster comparison approach not applicable here

---

## Comparison Group Strategy

For each non-SMA disease, analysis will be run against two comparison populations:

| Group | Description |
|---|---|
| Reference group 1 | All patients in DIZ data without that diagnosis |
| Reference group 2 (cluster) | Patients with clinically or lab-chemically similar differential diagnoses |

Steps 3–6 are conducted primarily against Reference group 2, as PPV is only clinically meaningful against a phenotypically similar differential diagnosis population.

---

## Infrastructure Milestones

- [ ] FDPG data access application approved
- [ ] Real DIZ data ingestion (FHIR queries via R `fhircrackr`)
- [ ] FLAME deployment to real DIZ nodes (Lane 2)
- [ ] Results from first real-data run reported
