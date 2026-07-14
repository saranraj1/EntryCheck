# Candidate dataset license and datasheet review

## Selection
Primary real-world set: **Adult, German Credit, Bank Marketing, and Breast Cancer Wisconsin Diagnostic**. **Spambase** is the reserve high-dimensional dataset.

All are distributed by UCI under **CC BY 4.0** with official DOIs. The study will preserve immutable snapshots, hashes, retrieval dates, and required citations.

| Dataset | DOI | Size | Features | Research role | Main caveat | Decision |
|---|---|---:|---:|---|---|---|
| Adult | 10.24432/C5XW20 | 48,842 | 14 | Mixed types, missing tokens, subgroup analysis | Historical 1994 sample and sensitive labels | Include |
| German Credit | 10.24432/C5NC77 | 1,000 | 20 | Small finance benchmark and subgroup diagnostics | High-stakes framing and low power | Include |
| Bank Marketing | 10.24432/C5K306 | 45,211 | 16 | Larger mixed-type and temporal dataset | Contact duration leakage; substantive unknown categories | Include with leakage-safe primary model |
| Breast Cancer Wisconsin Diagnostic | 10.24432/C5DW2B | 569 | 30 | Natural collinearity stress test | Small clinical dataset; no demographic groups | Include for correlation track |
| Spambase | 10.24432/C53G6X | 4,601 | 57 | Dimensionality/runtime sensitivity | Collection-specific non-spam indicators | Reserve |

## Preliminary datasheets

### Adult
Census-derived income classification. Treat question-mark tokens as missing. Sex and race support descriptive subgroup analysis, but the study must not claim current representativeness, causal fairness, or suitability for income decisions.

### German Credit
Classifies applicants as good/bad credit risks and includes an asymmetric cost matrix. Use sex/age groups only after power checks. Report uncertainty and avoid deployment claims.

### Bank Marketing
Predicts term-deposit subscription. Exclude `duration` from the primary model because it is known only after contact; optionally include it as a leakage sensitivity analysis. Prefer ordered data and add a temporal-split sensitivity check.

### Breast Cancer Wisconsin Diagnostic
Classifies malignant/benign masses from FNA-derived measurements. Radius, perimeter, and area families provide natural correlation. Use only as a benchmark; make no clinical recommendation.

### Spambase
Spam classification using word/character frequencies and capital-run statistics. Reserve for dimensionality. Collection-specific indicators limit generalization.

## Frozen preprocessing rules
1. Fit all preprocessing on training data only.
2. Preserve raw columns and deterministic feature lineage after encoding.
3. Group one-hot and deliberately redundant features for grouped attribution.
4. Separate native from induced missingness.
5. Publish snapshot hashes and retrieval dates.
6. Document subgroup provenance and category limitations.

## Official sources
- https://archive.ics.uci.edu/dataset/2/adult
- https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- https://archive.ics.uci.edu/dataset/222/bank+marketing
- https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- https://archive.ics.uci.edu/dataset/94/spambase
