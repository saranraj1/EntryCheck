# ExplainCheck preregistration-ready record

## Administration
- Title: ExplainCheck: Stress-Testing Tabular Feature Attribution under Correlation and Missingness
- Researcher: Saranraj U
- Study type: computational benchmark
- Human participants: none
- Status: ready for researcher verification and external timestamping

## Hypotheses
H1: No explainer dominates fidelity, stability, sparsity, and runtime.  
H2: Performance interacts with model and dataset.  
H3: Correlation reduces individual agreement more than grouped conservation.  
H4: Drift increases with missingness and depends on handling strategy among prediction-preserved cases.  
H5: At least one aggregate result conceals a practically meaningful subgroup gap.  
H6: At least one controlled model update passes predictive tolerance but fails an explanation gate.

## Design
The datasets, explainers, models, metrics, seeds, stressors, statistical plan, exclusions, and reproducibility rules are frozen in `PROTOCOL_V1.0.md`; deposit both files together.

## Confirmatory outcomes
Deletion fidelity AOPC; prediction-conditioned stability profile; k90 sparsity; runtime/resource use; correlation robustness; missingness sensitivity; subgroup consistency.

## Confirmatory/exploratory boundary
H1–H6 and primary metrics are confirmatory. Metric-correlation analyses, composite scores, MNAR, global/counterfactual explanations, and additional datasets/models are exploratory unless added before external registration.

## Stopping rule
Complete every prespecified dataset × model × explainer × seed cell or record a deterministic failure after the frozen retry policy. Do not stop based on apparent support for a hypothesis.

## AI disclosure
Notion AI/IGRIS assisted with literature synthesis, protocol drafting, code scaffolding, and documentation. Saranraj U must verify sources, execute and inspect code, approve decisions, and accept full responsibility. The AI is not an author.

## External-registration checklist
- [ ] Verify the Quantus/OpenXAI comparison.
- [ ] Verify every dataset license and citation.
- [ ] Approve Protocol v1.0.
- [ ] Hash the repository and protocol files.
- [ ] Separate pilot from confirmatory outputs.
- [ ] Deposit in OSF or another timestamped registry.
- [ ] Record the registration DOI/URL.
