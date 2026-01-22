# Property Prediction Datasets

This page describes the molecular property prediction datasets used in our benchmark, extracted from the [Polaris Hub](https://polarishub.io/). These datasets cover a wide range of ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties and target-specific bioactivity predictions.

## Overview

| Dataset                                            | Task Type | Property |
|----------------------------------------------------|-----------|----------|
| ASAP Discovery - MERS-CoV Mpro                   | Regression | Antiviral Potency |
| AstraZeneca LogD                                 | Regression | Lipophilicity |
| AstraZeneca PPB Clearance                        | Regression | Plasma Protein Binding |
| Novartis CYP3A4                                  | Regression | CYP Inactivation |
| PKIS2 Kinase Inhibition | Classification | Kinase Inhibition |
| Biogen ADME                | Regression | Various ADME Properties |
| TDCommons                    | Mixed | Various ADMET Properties |

---

## Assay description

<div class="grid cards" markdown>

-   :material-virus:{ .lg .middle } __ASAP Discovery - MERS-CoV Mpro__

    ---

    **Source:** `asap-discovery/antiviral-potency-2025-unblinded`

    **Task Type:** Regression

    **Target Property:** pIC50 against MERS-CoV main protease

    This dataset contains compounds tested for their inhibitory activity against the MERS-CoV main protease (Mpro), essential for viral replication. Main proteases are highly conserved across coronaviruses, making them attractive targets for broad-spectrum antiviral development.


-   :material-water:{ .lg .middle } __AstraZeneca LogD__

    ---

    **Source:** `polaris/az-logd-74-v1`

    **Task Type:** Regression

    **Target Property:** Octan-1-ol/water (pH 7.4) distribution coefficient (LogD)

    LogD at pH 7.4 measures lipophilicity under physiological conditions, accounting for ionization state. Optimal values (1-3) are associated with good oral bioavailability and CNS penetration.


-   :material-blood-bag:{ .lg .middle } __AstraZeneca PPB Clearance__

    ---

    **Source:** `polaris/az-ppb-clearance-v1`

    **Task Type:** Regression

    **Target Property:** Log percent of compound unbound to whole human plasma

    Plasma protein binding measures how much drug binds to plasma proteins (mainly albumin). Only the unbound fraction is pharmacologically active. High binding affects distribution, clearance, and drug-drug interactions.


-   :material-test-tube:{ .lg .middle } __Novartis CYP3A4__

    ---

    **Source:** `novartis/novartis-cyp3a4-v1`

    **Task Type:** Regression

    **Target Property:** Log-inactivation rate constant (log kobs) of CYP enzymes

    Measures time-dependent inhibition of CYP3A4, the most abundant liver enzyme metabolizing ~50% of marketed drugs. CYP3A4 inhibition can cause serious drug-drug interactions.


-   :material-target:{ .lg .middle } __PKIS2 - EGFR__

    ---

    **Source:** `polaris/drewry2017-pkis2-subset-v2`

    **Task Type:** Classification

    **Target Property:** Inhibitor of the EGFR kinase

    EGFR is a receptor tyrosine kinase involved in cell proliferation. EGFR inhibitors are used in cancer therapy for non-small cell lung cancer and colorectal cancer.

-   :material-target-account:{ .lg .middle } __PKIS2 - KIT__

    ---

    **Source:** `polaris/drewry2017-pkis2-subset-v2`

    **Task Type:** Classification

    **Target Property:** Inhibitor of the KIT kinase

    KIT plays a role in cell survival and proliferation. KIT inhibitors treat gastrointestinal stromal tumors (GIST) and certain leukemias.

-   :material-bullseye-arrow:{ .lg .middle } __PKIS2 - RET__

    ---

    **Source:** `polaris/drewry2017-pkis2-subset-v2`

    **Task Type:** Classification

    **Target Property:** Inhibitor of the RET kinase

    RET is involved in cell growth and differentiation. RET inhibitors are approved for RET-fusion positive cancers and medullary thyroid carcinoma.

-   :material-crosshairs:{ .lg .middle } __PKIS2 - LOK__

    ---

    **Source:** `polaris/drewry2017-pkis2-subset-v2`

    **Task Type:** Classification

    **Target Property:** Inhibitor of the LOK kinase

    LOK (STK10) is a serine/threonine kinase involved in lymphocyte migration and immune cell function.

-   :material-crosshairs-gps:{ .lg .middle } __PKIS2 - SLK__

    ---

    **Source:** `polaris/drewry2017-pkis2-subset-v2`

    **Task Type:** Classification

    **Target Property:** Inhibitor of the SLK kinase

    SLK is involved in cell cycle regulation, apoptosis, and cytoskeletal organization.


-   :material-water-opacity:{ .lg .middle } __Solubility__

    ---

    **Source:** `biogen/adme-fang-solu-reg-v1`

    **Task Type:** Regression

    **Target Property:** Log-solubility

    Aqueous solubility affects drug absorption and formulation. Poor solubility is a major cause of drug development failures.

-   :material-test-tube-off:{ .lg .middle } __Rat Plasma Protein Binding__

    ---

    **Source:** `biogen/adme-fang-rppb-reg-v1`

    **Task Type:** Regression

    **Target Property:** Log-rat plasma protein binding rate

    Fraction of compound bound to rat plasma proteins, useful for preclinical pharmacokinetic studies.

-   :material-human:{ .lg .middle } __Human Plasma Protein Binding__

    ---

    **Source:** `biogen/adme-fang-hppb-reg-v1`

    **Task Type:** Regression

    **Target Property:** Log-human plasma protein binding rate

    Fraction bound to human plasma proteins, critical for clinical pharmacokinetic predictions.

-   :material-transfer:{ .lg .middle } __Permeability (MDR1-MDCK)__

    ---

    **Source:** `biogen/adme-fang-perm-reg-v1`

    **Task Type:** Regression

    **Target Property:** Log-MDR1 MDCK efflux ratio

    Measures P-glycoprotein mediated efflux. P-gp substrates may have limited brain penetration and variable oral bioavailability.

-   :material-human-male:{ .lg .middle } __Human Liver Microsomal Stability__

    ---

    **Source:** `biogen/adme-fang-hclint-reg-v1`

    **Task Type:** Regression

    **Target Property:** Log-human liver microsomal stability (CLint)

    Intrinsic clearance predicts hepatic metabolic stability. Higher values indicate faster metabolism affecting exposure and dosing.

-   :material-microscope:{ .lg .middle } __Rat Liver Microsomal Stability__

    ---

    **Source:** `biogen/adme-fang-rclint-reg-v1`

    **Task Type:** Regression

    **Target Property:** Log-rat liver microsomal stability (CLint)

    Intrinsic clearance in rat liver microsomes for preclinical species selection and PK prediction.


-   :material-gate:{ .lg .middle } __P-glycoprotein Inhibition__

    ---

    **Source:** `tdcommons/pgp-broccatelli`

    **Task Type:** Classification

    **Target Property:** Inhibitor of P-glycoprotein (P-gp)

    P-gp is an efflux transporter. P-gp inhibitors can enhance brain penetration but may cause drug-drug interactions.

-   :material-brain:{ .lg .middle } __Blood-Brain Barrier Penetration__

    ---

    **Source:** `tdcommons/bbb-martins`

    **Task Type:** Classification

    **Target Property:** Ability to penetrate the blood-brain barrier (BBB)

    Critical for CNS drug development. BBB penetration is necessary for brain-targeting drugs but should be avoided for peripheral drugs.

-   :material-stomach:{ .lg .middle } __Caco-2 Permeability__

    ---

    **Source:** `tdcommons/caco2-wang`

    **Task Type:** Regression

    **Target Property:** Rate of compounds passing through Caco-2 cells

    Caco-2 cells model intestinal absorption. Low permeability often correlates with poor oral bioavailability.



-   :material-expand-all:{ .lg .middle } __Volume of Distribution__

    ---

    **Source:** `tdcommons/vdss-lombardo`

    **Task Type:** Regression

    **Target Property:** Volume of distribution at steady state (Vdss)

    High Vdss indicates extensive tissue distribution; low Vdss suggests the drug remains in plasma. Affects dosing and accumulation.

-   :material-clock-outline:{ .lg .middle } __Half-Life__

    ---

    **Source:** `tdcommons/half-life-obach`

    **Task Type:** Regression

    **Target Property:** Duration for drug concentration to be reduced by half

    Short half-life requires frequent dosing; long half-life may lead to accumulation.

-   :material-flask:{ .lg .middle } __Hepatocyte Clearance__

    ---

    **Source:** `tdcommons/clearance-hepatocyte-az`

    **Task Type:** Regression

    **Target Property:** Drug clearance measured in hepatocytes

    More physiologically relevant than microsomes as it includes Phase I and Phase II metabolism.

-   :material-flask-outline:{ .lg .middle } __Microsome Clearance__

    ---

    **Source:** `tdcommons/clearance-microsome-az`

    **Task Type:** Regression

    **Target Property:** Drug clearance measured in microsomes

    Primarily reflects CYP-mediated Phase I metabolism.

-   :material-water-percent:{ .lg .middle } __Lipophilicity__

    ---

    **Source:** `tdcommons/lipophilicity-astrazeneca`

    **Task Type:** Regression

    **Target Property:** Lipophilicity (LogD)

    Affects membrane permeability, protein binding, metabolism, and overall pharmacokinetics.


-   :material-alert-octagon:{ .lg .middle } __Drug-Induced Liver Injury (DILI)__

    ---

    **Source:** `tdcommons/dili`

    **Task Type:** Classification

    **Target Property:** Potential to induce liver injuries

    DILI is a major cause of drug withdrawal. Hepatotoxicity is a leading cause of clinical trial failures.

-   :material-heart-pulse:{ .lg .middle } __hERG Inhibition__

    ---

    **Source:** `tdcommons/herg`

    **Task Type:** Classification

    **Target Property:** Blocker of hERG channel

    hERG inhibition can cause QT prolongation and fatal cardiac arrhythmias. Screening is mandatory in drug development.

-   :material-dna:{ .lg .middle } __Ames Mutagenicity__

    ---

    **Source:** `tdcommons/ames`

    **Task Type:** Classification

    **Target Property:** Mutagenic potential (Ames test positive/negative)

    Mutagenic compounds are potential carcinogens. Positive Ames test often disqualifies compounds from development.

-   :material-skull:{ .lg .middle } __Acute Toxicity (LD50)__

    ---

    **Source:** `tdcommons/ld50-zhu`

    **Task Type:** Regression

    **Target Property:** Acute toxicity LD50 (lethal dose for 50% of test animals)

    Provides initial safety assessment and helps establish safe starting doses.


-   :material-pill:{ .lg .middle } __CYP2C9 Substrate__

    ---

    **Source:** `tdcommons/cyp2c9-substrate-carbonmangels`

    **Task Type:** Classification

    **Target Property:** Substrate of CYP2C9

    CYP2C9 metabolizes ~15% of drugs including warfarin and NSAIDs.

-   :material-medication:{ .lg .middle } __CYP2D6 Substrate__

    ---

    **Source:** `tdcommons/cyp2d6-substrate-carbonmangels`

    **Task Type:** Classification

    **Target Property:** Substrate of CYP2D6

    CYP2D6 is highly polymorphic, affecting ~25% of drugs. Genetic variations lead to poor, intermediate, extensive, and ultra-rapid metabolizer phenotypes.

-   :material-medical-bag:{ .lg .middle } __CYP3A4 Substrate__

    ---

    **Source:** `tdcommons/cyp3a4-substrate-carbonmangels`

    **Task Type:** Classification

    **Target Property:** Substrate of CYP3A4

    CYP3A4 is the most important drug-metabolizing enzyme, processing ~50% of drugs.



-   :material-beaker:{ .lg .middle } __Solubility (AqSolDB)__

    ---

    **Source:** `tdcommons/solubility-aqsoldb`

    **Task Type:** Regression

    **Target Property:** Aqueous solubility

    Data from AqSolDB, one of the largest curated aqueous solubility datasets. Essential for drug absorption and formulation.

</div>

---

## Reward Functions

Property prediction tasks use different reward functions depending on whether the task is regression or classification:

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } __Regression Reward__

    ---

    $$R = 1 - \frac{(\hat{y} - y)^2}{\sigma^2}$$

    Where $\hat{y}$ is the predicted value, $y$ is the ground truth, and $\sigma$ is the standard deviation of training labels. This normalizes prediction errors and ensures rewards are in [0, 1].

    **Training samples:** ~44,000

-   :material-check-circle:{ .lg .middle } __Classification Reward__

    ---

    $$R = \mathbb{1}_{pred = label}$$

    Binary reward: 1 if the prediction exactly matches the ground truth label, 0 otherwise.

    **Training samples:** ~11,000

</div>

!!! warning "Invalid Predictions"
    If the model generates an invalid or unparseable prediction, the reward is automatically set to 0.

---

## References

1. ASAP Discovery Consortium. Antiviral Potency Dataset (2025).
2. Drewry, D.H., et al. "Progress towards a public chemogenomic set for protein kinases and a call for contributions." *PLOS ONE* (2017).
3. Lombardo, F., et al. "Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters in Humans for 670 Drug Compounds." *Drug Metabolism and Disposition* (2008).
4. Martins, I.F., et al. "A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling." *Journal of Chemical Information and Modeling* (2012).
5. Therapeutics Data Commons: https://tdcommons.ai/
6. Polaris Hub: https://polarishub.io/
