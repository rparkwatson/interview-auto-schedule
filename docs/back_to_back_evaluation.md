# Back-to-back avoidance evaluation

## Scope
This evaluation reviews how back-to-back avoidance is currently implemented in the scheduler and what practical behavior it creates.

## What the feature currently does

1. The UI exposes three modes: `soft`, `hard`, and `off`.
2. Adjacency is computed per day and only between a slot and its immediate next slot when the gap is 0 minutes (or within optional grace).
3. In `hard` mode, an interviewer cannot be assigned to both adjacent slots.
4. In `soft` mode, each adjacent pair for an interviewer creates a boolean penalty indicator, and the objective subtracts `w_b2b * total_b2b`.
5. In `off` mode, no adjacency constraints/penalties are added.

## Functional observations

### Strengths
- `hard` mode behaves as expected and strictly blocks adjacent assignments.
- `soft` mode correctly measures adjacency events (`b2b` variables are populated and returned).
- The adjacency graph is deterministic and simple.

### Gaps / risks

1. **Soft mode is effectively weak under current default weights.**
   - `w_pairs` defaults to `1,000,000` while `w_b2b` defaults to `1`.
   - In the current UI, objective weights are disabled and `w_b2b` input is disabled as well.
   - Net effect: soft penalties can only break ties between equal-fill solutions; they cannot realistically trade off small room-fill reductions for materially better spacing.

2. **Adjacency only looks one slot ahead.**
   - This catches strict back-to-back pairs but does not address broader run-shape problems (e.g., three in a row beyond pairwise adjacency treatment quality).

3. **No role-specific policy.**
   - Regular, Senior, and Observer interviewers are treated identically for adjacency penalties/constraints.

4. **Limited explainability in UI reporting.**
   - The model returns `b2b`, but users are not given a concise summary like "who has the most consecutive assignments" by day.

## Quick behavior check run

A small synthetic run (2 Regular interviewers, 3 contiguous slots, capacity 1 per slot) showed:

- `off`: fills all 3 slots (3 pairs) with full consecutive assignments.
- `soft` (default weights): also fills all 3 slots and keeps 4 adjacency violations.
- `hard`: drops to 2 pairs and removes adjacency violations.

This confirms that soft mode currently works mostly as a tie-breaker, not a meaningful spacing optimizer.

## Suggested improvements (priority order)

1. **Introduce lexicographic optimization stages** (recommended).
   - Stage 1: maximize pairs/fill.
   - Stage 2: with fill fixed, minimize back-to-back and fairness penalties.
   - This preserves capacity outcomes while making spacing improvements predictable.

2. **Enable/tune spacing controls in UI.**
   - Re-enable `w_b2b` and related objective controls, or provide a simple "spacing aggressiveness" setting mapped to safe coefficients.

3. **Add run-shape constraints beyond adjacent pairs.**
   - Add optional soft/hard "no 3 in a row" and/or day-level spacing targets.

4. **Expand adjacency graph options.**
   - Optionally treat near-adjacent short-gap slots as consecutive with configurable thresholds by role.

5. **Add reporting diagnostics.**
   - Show per-interviewer consecutive counts and top offenders in results/exports.

6. **Consider role-specific spacing policy.**
   - Different penalties for Regular vs Senior/Observer, if operations need this.
