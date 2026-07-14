# Publication strategy

## Decision: paired artifact strategy
The **primary output is an empirical benchmark paper** supported by executable open-source research software. After the API and documentation stabilize, prepare a **separate software paper** without duplicating the benchmark paper’s empirical claims.

## Rationale
- The intended novelty is methodological and empirical: stress protocols, prediction-conditioned stability, and evidence-to-governance outputs.
- A software-only paper would not answer the research questions.
- A single oversized paper could dilute both the experimental and engineering contributions.

## Contribution order
1. Stress-test protocol and construct definitions.
2. Findings across explainers, models, datasets, and stressors.
3. Uncertainty and metric disagreement.
4. Reproducible model-card/CI bridge.

## Release sequence
1. Protocol and preregistration package.
2. Public v0.1 research code and synthetic pilot.
3. Frozen confirmatory artifact with DOI.
4. Benchmark paper submission.
5. Software-paper submission after API stability and evidence of reuse.

## Venue families
Investigate XAI workshops, AAAI/ACM AIES, applied AI/ML journals accepting benchmark papers, and datasets-and-benchmarks tracks. Consider FAccT only if the socio-technical analysis becomes sufficiently deep.
