# Use case

## The decision
*Which drug candidates should a pharmaceutical company fund this cycle to get the most risk-adjusted value, without betting too much on drugs that tend to fail together?*

- **Variables:** one yes/no per candidate (fund or not)
- **Objective:** maximize Σ (probability of approval × peak value − cost), minus a penalty for correlated risk (candidates in the same therapeutic area fail together)
- **Constraint:** fund exactly *k* candidates (capacity/budget), added as the penalty P·(Σx − k)²
- **Math:** Markowitz portfolio selection applied to healthcare assets, the same structure used to pick stock portfolios

## Who it affects
Pharma R&D portfolio committees, biotech investors, and ultimately patients: an unfunded candidate is a therapy that never reaches them. Most candidates entering clinical trials are never approved, and late-stage trials cost hundreds of millions of dollars, so better portfolio decisions are worth well over $10M a year to a mid-size company.

## How it's done today
Committees combine expert judgment with spreadsheet scoring, and sometimes classical optimizers (mixed-integer programming, heuristics). At realistic sizes (50–200 candidates), classical solvers work well. Quantum methods are not yet competitive, and this project measures by how much.

## Why more than one industry
The same QUBO structure appears in finance (stock portfolios), healthcare (choosing a small, non-redundant set of diagnostic tests), and networks (Max-Cut). Q-Router tests whether knowledge about *how to solve* one field's problems transfers to another's: which quantum method fits, and even the tuned circuit settings.

## What it would take to be real
Real pipeline data (e.g. from ClinicalTrials.gov and company disclosures), problem sizes of 50–200 variables, and quantum hardware with roughly 10–100× lower two-qubit error rates, or qubit-compression encodings, before quantum methods could compete with classical solvers.
