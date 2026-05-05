# quantum-inspired-resource-allocation-qubo-sa-cpu

Quantum-inspired solver for the **Resource Allocation for Maintenance Depots
(TEKSEM)** use case.

## Approach

The MILP from the classical baseline is reformulated as a **QUBO** (Quadratic
Unconstrained Binary Optimisation) suitable for variational quantum algorithms
(VQA / QAOA), digital annealers, or quantum-inspired samplers. We solve the
resulting Ising hamiltonian with **simulated annealing** as a faithful, no-network
proxy for what a near-term VQA would produce on the same problem mapping.

Variables are one binary `x[task, technician]` per compatible pair plus one
slack `u[task]` indicating "task left unassigned":

```
H =  w_obj  · Σ_{t,r} cost[t,r] · x[t,r]
   + w_un   · Σ_t u[t]
   + λ_a    · Σ_t (Σ_r x[t,r] + u[t] − 1)²        # 1-of-K with slack
   + λ_c    · Σ_r ReLU(Σ_t dur[t]·x[t,r] − cap[r])²
   + λ_s    · Σ_p ReLU(Σ_t dem[t,p]·(1−u[t]) − stock[p])²
```

Output is decoded into the same canonical schema as the classical solver
(`objective_value` in EUR), so the two are directly benchmarkable on the
QCentroid platform.

## solver_params

- `max_iterations` *(int, default 1500)* — outer SA temperature steps.
- `inner_steps` *(int, default `max(20, num_qubits)`)* — Metropolis steps per temperature.
- `initial_temperature` *(float, default ≈ avg pair-cost)* — `T0`.
- `final_temperature` *(float, default 1e-3)* — `Tf` (geometric schedule).
- `seed` *(int, default 42)* — RNG seed for reproducibility.
- `lambda_assignment`, `lambda_capacity`, `lambda_stock` *(float)* — penalty weights;
  defaults auto-scale to the average objective coefficient.

## Additional outputs (vs. classical)

The output dict adds `additional_output` and extends `computation_metrics` with:
- `num_qubits` — size of the QUBO / Ising problem.
- `energy_curve` — best energy per outer SA step (renderable as a convergence chart).
- `sa_temperature_schedule` — `{T0, Tf, n_outer, n_inner}`.
- `feasibility_breakdown` — `{assignment_violation, capacity_violation, stock_violation}`
  to expose how well the soft constraints converged.
- `additional_output.qubo_size` and `additional_output.penalty_weights` — for the
  QCentroid additional-output viewer.

## Notes

- Comparable units (`objective_value` in EUR) with
  `classical-resource-allocation-milp-cpu`. Direct benchmark via `qc_compare_jobs`.
- Hardware-portable: the same QUBO can be sampled on D-Wave (`dwave-ocean-sdk`)
  or compiled to a QAOA ansatz on Qiskit / CUDA-Q with no further
  problem-specific code.
