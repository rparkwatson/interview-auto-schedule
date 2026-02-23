# Scheduling model improvement plan

This document summarizes the expected impact of the remaining recommended changes and proposes additional enhancements to improve schedule quality and operability.

## 1) Lexicographic / multi-pass objective

### Current state
The solver uses a single weighted objective that blends room fill, adcom fill, scarcity-weighted regular assignment, back-to-back penalties, fairness spread penalties, and tiny tie-break jitter.

### Proposed approach
Use staged optimization (lexicographic policy):
1. maximize regular pairs
2. fix regular pairs at optimum and maximize total rooms used / adcom singles
3. fix step 1-2 values and minimize fairness penalties + back-to-back
4. keep tiny seeded jitter for deterministic tie-breaking only

### Impact
- More predictable tradeoffs (fewer weight-tuning surprises)
- Easier business explanation of outcomes
- Better stability when adding future soft constraints
- Slightly longer solve time due to multiple solves per run

### Likely touched files
- `scheduler/solvers/cpsat.py`
- `scheduler/config.py`
- `pages/02_Scheduler.py` (if exposing strategy toggle)

## 2) Better infeasibility diagnostics in UI/reporting

### Current state
The UI reports solver status and suppresses misleading metrics on unsolved runs, but root-cause visibility is limited.

### Proposed approach
Add a diagnostics pass when status is infeasible/unknown:
- report aggregate impossibility checks pre-solve (e.g., required mins > available capacity)
- add optional soft-relax model with violation vars per rule family (min totals, daily caps, room caps)
- render a ranked "top blockers" panel and include in exports

### Impact
- Faster iterative tuning by non-technical users
- Lower support burden (clear reasons instead of generic infeasible)
- Better confidence that policy settings are realistic

### Likely touched files
- `pages/02_Scheduler.py`
- `scheduler/solvers/cpsat.py`
- `scheduler/io/write.py`

## 3) Observer ingestion enhancement

### Current state
Domain + solver support observer logic, but legacy reader currently only ingests Regular and Senior sheets.

### Proposed approach
Add observer source support in legacy parsing (new optional sheet name and defaults), and expose observer constraints in UI.

### Impact
- Aligns ingestion with solver capabilities already present
- Enables full use of observer_extra_per_slot setting
- Minimal operational risk if kept optional and backward-compatible

### Likely touched files
- `scheduler/io/read.py`
- `pages/02_Scheduler.py`
- (optionally) workbook builder pages/templates

## Additional high-value improvements

1. **Distribution-aware fairness**
   - Move beyond spread (max-min) to target-based or convex-like balancing (e.g., minimize sum of absolute deviations from target load).
2. **Interviewer preference modeling**
   - Add preference scores per slot and maximize weighted preference satisfaction after fill constraints.
3. **No-three-in-a-row and day-shape controls**
   - Penalize long runs and reward better spacing, not just adjacent back-to-back.
4. **Scenario ensemble + robustness report**
   - Run multiple seeds and summarize variance (who is consistently overloaded/underutilized).
5. **Post-solve local search polish**
   - Short hill-climb swap phase to improve fairness/spacing while preserving primary metrics.
6. **Data quality gates**
   - Pre-solve checks for malformed slot labels, duplicate interviewer IDs, and impossible min/max settings.
7. **Explainability export**
   - Include why each assignment happened (constraint class + objective contribution summary) in Excel report.

## Suggested rollout order
1. Infeasibility diagnostics (highest UX value)
2. Lexicographic objective strategy (highest model governance value)
3. Observer ingestion (if operations use observers)
4. Advanced fairness + preference model extensions
