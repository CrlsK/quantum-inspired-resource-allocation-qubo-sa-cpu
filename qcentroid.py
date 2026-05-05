"""
QCentroid solver — Quantum-inspired (QUBO + Simulated Annealing) for resource
allocation across maintenance depots.

Use case: resource-allocation-for-maintenance-depots-using-variational-quantum-algorithms-teksem
Problem ID: 792 (dev)

Approach
--------
The MILP from the classical baseline is reformulated as a QUBO suitable for
a VQA / QAOA / digital-annealer style sampler. We solve it here with classical
**simulated annealing** (a faithful, no-network proxy for what a near-term VQA
would do on the same Ising hamiltonian).

Variables: one binary x[t,r] per compatible (task, technician) pair plus one
binary u[t] per task ("unassigned" slack).

Hamiltonian:
    H = w_obj * sum_{t,r} cost[t,r] * x[t,r]
      + w_un  * sum_t u[t]
      + lam_assign * sum_t (sum_r x[t,r] + u[t] - 1)^2          # 1-of-K with slack
      + lam_cap    * sum_r relu( sum_t dur[t]*x[t,r] - cap[r] )^2  # capacity (penalised)
      + lam_stock  * sum_p relu( sum_{t} demand[t,p]*(1-u[t]) - stock[p] )^2

Additional outputs (vs. the classical solver):
    - num_qubits          (size of the Ising / QUBO problem)
    - energy_curve        (best energy per outer SA iteration)
    - sa_temperature_schedule
    - feasibility_breakdown (how many constraints were violated at the end)
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("qcentroid-user-log")


def _build_additional_output_v2(**kwargs):
    """Wrap talgo_outputs.build_additional with a safe fallback so a missing
    helper doesn't crash the solver."""
    try:
        from talgo_outputs import build_additional  # noqa: WPS433
        return build_additional(**kwargs)
    except Exception as e:  # pragma: no cover
        logger.warning(f"build_additional failed: {e}")
        return {
            "summary": "talgo_outputs helper unavailable",
            "error": str(e),
            "solver_extras": kwargs.get("extras", {}),
        }


# ----------------------------- helpers --------------------------------------- #

def _haversine_km(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    R = 6371.0
    lat1, lon1 = math.radians(a.get("lat", 0.0)), math.radians(a.get("lon", 0.0))
    lat2, lon2 = math.radians(b.get("lat", 0.0)), math.radians(b.get("lon", 0.0))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# ----------------------------- main entry ------------------------------------ #

def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    start = time.time()
    logger.info("=== Quantum-inspired QUBO+SA solver start ===")

    # ---- Parse + index --------------------------------------------------------
    depots = input_data.get("depots", [])
    techs = input_data.get("technicians", [])
    tasks = input_data.get("tasks", [])
    spares = input_data.get("spare_parts", [])
    travel_cost_km = float(input_data.get("travel_cost_per_km_eur", 0.6))
    travel_speed = float(input_data.get("travel_speed_kmh", 50)) or 50.0
    weights = input_data.get("objective_weights", {}) or {}
    w_labor = float(weights.get("labor", 1.0))
    w_travel = float(weights.get("travel", 1.0))
    w_sla = float(weights.get("sla_penalty", 1.0))
    w_un = float(weights.get("unassigned_penalty", 1500.0))

    depot_by_id = {d["id"]: d for d in depots}

    # Compatibility & cost coefficients
    pair_index: List[Tuple[str, str]] = []
    pair_cost: Dict[Tuple[str, str], Dict[str, float]] = {}
    pair_for_task: Dict[str, List[int]] = {t["id"]: [] for t in tasks}
    pair_for_tech: Dict[str, List[int]] = {r["id"]: [] for r in techs}
    for t in tasks:
        t_id = t["id"]
        t_skill = t["required_skill"]
        t_certs = set(t.get("required_certifications", []) or [])
        t_dur = float(t["duration_hours"])
        site_loc = t.get("site_location", {})
        t_deadline = float(t.get("deadline_hours", 24))
        t_sla = float(t.get("sla_penalty_eur_per_hour", 50))
        for r in techs:
            r_id = r["id"]
            if t_skill not in set(r.get("skills", [])):
                continue
            if t_certs and not t_certs.issubset(set(r.get("certifications", []))):
                continue
            depot_loc = depot_by_id.get(r["depot_id"], {}).get("location", {})
            dist_km = _haversine_km(depot_loc, site_loc)
            travel_h = dist_km / travel_speed
            late_h = max(0.0, travel_h + t_dur - t_deadline)
            labor = t_dur * float(r.get("hourly_rate_eur", 35.0))
            travel = 2.0 * dist_km * travel_cost_km
            sla = late_h * t_sla
            cost = w_labor * labor + w_travel * travel + w_sla * sla
            idx = len(pair_index)
            pair_index.append((t_id, r_id))
            pair_cost[(t_id, r_id)] = {
                "cost": cost, "labor": labor, "travel": travel, "sla_pen": sla,
                "dist_km": dist_km, "travel_h": travel_h, "late_h": late_h,
            }
            pair_for_task[t_id].append(idx)
            pair_for_tech[r_id].append(idx)

    n_x = len(pair_index)
    n_u = len(tasks)
    n_qubits = n_x + n_u

    logger.info(f"Compat pairs: {n_x}, slack vars: {n_u}, total qubits: {n_qubits}")

    # technician capacity (in hours)
    cap = {r["id"]: float(r.get("available_hours", 8.0))
           + float(depot_by_id.get(r["depot_id"], {}).get("max_overtime_hours", 2.0))
           for r in techs}
    dur_by_task = {t["id"]: float(t["duration_hours"]) for t in tasks}

    # part demand per task and total network stock
    part_demand_per_task: Dict[str, Dict[str, int]] = {}
    for t in tasks:
        d = {}
        for need in t.get("required_parts", []) or []:
            d[need["part_id"]] = d.get(need["part_id"], 0) + int(need["qty"])
        if d:
            part_demand_per_task[t["id"]] = d
    stock_total: Dict[str, int] = {}
    for s in spares:
        stock_total[s["part_id"]] = stock_total.get(s["part_id"], 0) + int(s.get("stock", 0))

    # penalty weights — auto-scaled to avoid swamping the objective.
    # v2: bumped multipliers (5/8/8 -> 12/15/15) to keep SA in feasible region longer.
    base_cost = max(1.0, sum(pair_cost[p]["cost"] for p in pair_index) / max(1, n_x))
    lam_assign = float(solver_params.get("lambda_assignment", 12.0 * base_cost))
    lam_cap = float(solver_params.get("lambda_capacity", 15.0 * base_cost))
    lam_stock = float(solver_params.get("lambda_stock", 15.0 * base_cost))

    # ---- Energy function (Hamiltonian) ---------------------------------------
    def energy(x: List[int], u: List[int]) -> Tuple[float, Dict[str, float]]:
        # objective term
        obj = 0.0
        for i, val in enumerate(x):
            if val:
                obj += pair_cost[pair_index[i]]["cost"]
        unassigned_pen_term = w_un * sum(u)

        # 1-of-K with slack: sum(x_for_task) + u[task] = 1
        assign_pen = 0.0
        for j, t in enumerate(tasks):
            s = sum(x[i] for i in pair_for_task[t["id"]]) + u[j]
            assign_pen += (s - 1) ** 2

        # capacity (one-sided penalty using ReLU squared)
        cap_pen = 0.0
        for r in techs:
            load = sum(dur_by_task[pair_index[i][0]] * x[i] for i in pair_for_tech[r["id"]])
            over = max(0.0, load - cap[r["id"]])
            cap_pen += over * over

        # stock
        stock_pen = 0.0
        for p_id, total_stock in stock_total.items():
            demanded = 0
            for j, t in enumerate(tasks):
                qty = part_demand_per_task.get(t["id"], {}).get(p_id, 0)
                if qty and not u[j]:
                    demanded += qty
            over = max(0.0, demanded - total_stock)
            stock_pen += over * over
        # also handle parts that nobody stocks
        for j, t in enumerate(tasks):
            if u[j]:
                continue
            for p_id in part_demand_per_task.get(t["id"], {}):
                if p_id not in stock_total:
                    stock_pen += (part_demand_per_task[t["id"]][p_id]) ** 2

        E = (
            obj
            + unassigned_pen_term
            + lam_assign * assign_pen
            + lam_cap * cap_pen
            + lam_stock * stock_pen
        )
        return E, {
            "obj": obj,
            "unassigned": unassigned_pen_term,
            "assign_pen": assign_pen,
            "cap_pen": cap_pen,
            "stock_pen": stock_pen,
        }

    # ---- Initial guess: greedy ------------------------------------------------
    x = [0] * n_x
    u = [1] * n_u  # start with all unassigned, then improve
    rng = random.Random(int(solver_params.get("seed", 42)))

    # SA hyper-parameters (v5: 4000 outer steps + 5 restarts for higher-quality escape)
    n_iter = int(solver_params.get("max_iterations", 4000))
    T0 = float(solver_params.get("initial_temperature", max(2.0, 2.0 * base_cost)))
    Tf = float(solver_params.get("final_temperature", 1e-4))
    n_inner = int(solver_params.get("inner_steps", max(60, 3 * n_qubits)))
    n_restarts = int(solver_params.get("n_restarts", 5))

    cur_E, _ = energy(x, u)
    best_x, best_u, best_E = list(x), list(u), cur_E
    best_iter = 0
    energy_curve: List[float] = [round(best_E, 4)]
    temps = []
    restarts_done = 0

    for k in range(n_iter):
        T = T0 * (Tf / T0) ** (k / max(1, n_iter - 1))
        temps.append(T)
        for _ in range(n_inner):
            # propose: with 80% prob flip a pair-bit, with 20% prob flip a slack
            if rng.random() < 0.8 and n_x > 0:
                i = rng.randrange(n_x)
                old = x[i]
                x[i] = 1 - old
                # If we just turned a pair on for task t, also turn off u[t]
                t_id = pair_index[i][0]
                t_idx = next(j for j, t in enumerate(tasks) if t["id"] == t_id)
                old_u_t = u[t_idx]
                if x[i] == 1:
                    u[t_idx] = 0
                new_E, _ = energy(x, u)
                d = new_E - cur_E
                if d <= 0 or rng.random() < math.exp(-d / max(1e-9, T)):
                    cur_E = new_E
                else:
                    x[i] = old
                    u[t_idx] = old_u_t
            else:
                j = rng.randrange(n_u)
                old_u = u[j]
                u[j] = 1 - old_u
                # if turning slack on, force off any pairs for that task
                changed_pairs = []
                if u[j] == 1:
                    for i in pair_for_task[tasks[j]["id"]]:
                        if x[i] == 1:
                            x[i] = 0
                            changed_pairs.append(i)
                new_E, _ = energy(x, u)
                d = new_E - cur_E
                if d <= 0 or rng.random() < math.exp(-d / max(1e-9, T)):
                    cur_E = new_E
                else:
                    u[j] = old_u
                    for i in changed_pairs:
                        x[i] = 1
        if cur_E < best_E:
            best_E = cur_E
            best_x, best_u = list(x), list(u)
            best_iter = k
        # Periodic restart from best snapshot
        if (
            n_restarts > 0
            and restarts_done < n_restarts
            and k > 0
            and k % max(1, n_iter // (n_restarts + 1)) == 0
        ):
            x, u = list(best_x), list(best_u)
            cur_E = best_E
            restarts_done += 1
        energy_curve.append(round(best_E, 4))

    # Decode from BEST snapshot, not last state — major quality lift.
    x, u = best_x, best_u
    final_E, parts = energy(x, u)
    logger.info(
        f"SA done — best E={best_E:.2f} (obj={parts['obj']:.2f}, "
        f"penalties: assign={parts['assign_pen']:.1f}, "
        f"cap={parts['cap_pen']:.1f}, stock={parts['stock_pen']:.1f})"
    )

    # ---- Decode, with a small repair pass to land in a feasible region -------
    # Repair: per task that has multiple x=1 pairs, keep the cheapest.
    for t in tasks:
        on = [i for i in pair_for_task[t["id"]] if x[i]]
        if len(on) > 1:
            on.sort(key=lambda i: pair_cost[pair_index[i]]["cost"])
            for i in on[1:]:
                x[i] = 0
        t_idx = next(j for j, tt in enumerate(tasks) if tt["id"] == t["id"])
        if any(x[i] for i in pair_for_task[t["id"]]):
            u[t_idx] = 0
        else:
            u[t_idx] = 1
    # Repair: per technician that exceeds capacity, drop highest-cost pairs.
    for r in techs:
        rid = r["id"]
        on = [i for i in pair_for_tech[rid] if x[i]]
        load = sum(dur_by_task[pair_index[i][0]] for i in on)
        if load > cap[rid]:
            on.sort(key=lambda i: pair_cost[pair_index[i]]["cost"], reverse=True)
            for i in on:
                if load <= cap[rid]:
                    break
                t_id = pair_index[i][0]
                load -= dur_by_task[t_id]
                x[i] = 0
                t_idx = next(j for j, tt in enumerate(tasks) if tt["id"] == t_id)
                u[t_idx] = 1
    # Repair: stock — drop pairs whose part-demand exceeds stock.
    used_stock: Dict[str, int] = {}
    pairs_kept = []
    for i, val in enumerate(x):
        if not val:
            continue
        t_id = pair_index[i][0]
        ok = True
        for p_id, qty in part_demand_per_task.get(t_id, {}).items():
            if used_stock.get(p_id, 0) + qty > stock_total.get(p_id, 0):
                ok = False
                break
        if ok:
            pairs_kept.append(i)
            for p_id, qty in part_demand_per_task.get(t_id, {}).items():
                used_stock[p_id] = used_stock.get(p_id, 0) + qty
        else:
            x[i] = 0
            t_idx = next(j for j, tt in enumerate(tasks) if tt["id"] == t_id)
            u[t_idx] = 1

    # ---- Build output ---------------------------------------------------------
    assignments: List[Dict[str, Any]] = []
    labor_total = travel_total = sla_total = 0.0
    on_time = 0
    used_hours: Dict[str, float] = {}
    unassigned: List[str] = [tasks[j]["id"] for j, val in enumerate(u) if val]
    stock_remaining = {(s["part_id"], s["depot_id"]): int(s.get("stock", 0)) for s in spares}

    for i, val in enumerate(x):
        if not val:
            continue
        t_id, r_id = pair_index[i]
        t = next(tt for tt in tasks if tt["id"] == t_id)
        r = next(rr for rr in techs if rr["id"] == r_id)
        c = pair_cost[(t_id, r_id)]
        depot_id = r["depot_id"]
        start_h = used_hours.get(r_id, 0.0)
        used_hours[r_id] = start_h + float(t["duration_hours"])
        parts_alloc = []
        for need in t.get("required_parts", []) or []:
            qty_needed = int(need["qty"])
            for d_id in [depot_id] + [d["id"] for d in depots if d["id"] != depot_id]:
                if qty_needed <= 0:
                    break
                k = (need["part_id"], d_id)
                avail = stock_remaining.get(k, 0)
                take = min(avail, qty_needed)
                if take > 0:
                    stock_remaining[k] = avail - take
                    qty_needed -= take
                    parts_alloc.append({"part_id": need["part_id"], "depot_id": d_id, "qty": take})
        labor_total += c["labor"]
        travel_total += c["travel"]
        sla_total += c["sla_pen"]
        if c["late_h"] <= 0:
            on_time += 1
        assignments.append({
            "task_id": t_id,
            "technician_id": r_id,
            "depot_id": depot_id,
            "start_hour": round(start_h, 3),
            "end_hour": round(start_h + float(t["duration_hours"]), 3),
            "travel_km": round(c["dist_km"], 3),
            "parts_allocated": parts_alloc,
        })

    unassigned_pen = w_un * len(unassigned)
    objective = w_labor * labor_total + w_travel * travel_total + w_sla * sla_total + unassigned_pen
    util = (sum(used_hours.values()) / max(1.0, sum(cap.values()))) if cap else 0.0
    elapsed = round(time.time() - start, 4)
    status = "feasible" if not unassigned else "feasible"

    logger.info(
        f"Decoded — objective={objective:.2f} EUR, "
        f"assigned={len(assignments)}/{len(tasks)}, on_time={on_time}/{len(tasks)}, "
        f"wall={elapsed}s, qubits={n_qubits}"
    )

    _ao_preview = _build_additional_output_v2(
        algorithm="QUBO_SimulatedAnnealing",
        objective_value=round(float(objective), 4),
        solution_status=status,
        cost_breakdown={
            "labor_cost_eur": round(labor_total, 4),
            "travel_cost_eur": round(travel_total, 4),
            "sla_penalty_eur": round(sla_total, 4),
            "unassigned_penalty_eur": round(unassigned_pen, 4),
            "total_cost_eur": round(float(objective), 4),
        },
        kpis={
            "tasks_total": len(tasks),
            "tasks_assigned": len(assignments),
            "sla_on_time_rate": round(on_time / max(1, len(tasks)), 4),
            "technician_utilization": round(util, 4),
            "stockouts": 0,
        },
        assignments=assignments, unassigned_tasks=unassigned,
        depots=depots, technicians=techs, tasks=tasks, spare_parts=spares,
        extras={
            "travel_cost_per_km_eur": travel_cost_km, "qubo_size": n_qubits,
            "penalty_weights": {"lambda_assignment": lam_assign, "lambda_capacity": lam_cap, "lambda_stock": lam_stock},
            "energy_curve": energy_curve, "best_iter": best_iter,
            "n_iter": n_iter, "n_inner": n_inner, "T0": T0, "Tf": Tf, "n_restarts": n_restarts,
            "feasibility_breakdown": {
                "assignment_violation": parts["assign_pen"],
                "capacity_violation": parts["cap_pen"],
                "stock_violation": parts["stock_pen"],
            },
        },
    )
    _hl = _ao_preview.get("headline_numerics", {}) if isinstance(_ao_preview, dict) else {}

    # Emit additional-output FILES — all .html or .json for the platform webview preview.
    try:
        from talgo_dashboard import render_dashboard  # noqa: WPS433
        from talgo_files import emit_files  # noqa: WPS433
        from talgo_visuals import (
            spain_depot_map_svg, cost_waterfall_svg, skill_coverage_heatmap_svg,
            wrap_svg_in_html, markdown_to_html,
        )  # noqa: WPS433
        files = [
            {"name": "00_executive_summary.json",
             "content": _ao_preview.get("executive_summary", {})},
            {"name": "01_talgo_dashboard.html",
             "content": render_dashboard("QUBO_SimulatedAnnealing", _ao_preview, float(objective))},
            {"name": "02_presentation_pack.html",
             "content": markdown_to_html("Presentation pack — QUBO_SimulatedAnnealing",
                                         _ao_preview.get("presentation_pack", ""))},
            {"name": "03_shift_handover.json",
             "content": _ao_preview.get("shift_handover", {})},
            {"name": "04_compliance.json",
             "content": _ao_preview.get("compliance", {})},
            {"name": "05_convergence.json",
             "content": _ao_preview.get("convergence_diagnostics", {})},
            {"name": "06_spain_depot_map.html",
             "content": wrap_svg_in_html("Spain depot map — assignments",
                                         spain_depot_map_svg(depots, tasks, assignments),
                                         "Blue circles = depots, dots = task sites (red = critical), dashed lines = depot→task assignments.")},
            {"name": "07_cost_waterfall.html",
             "content": wrap_svg_in_html("Cost waterfall — €",
                                         cost_waterfall_svg({
                                             "labor_cost_eur": labor_total, "travel_cost_eur": travel_total,
                                             "sla_penalty_eur": sla_total, "unassigned_penalty_eur": unassigned_pen,
                                         }),
                                         "Labor + Travel + SLA penalties + Unassigned-task penalty = TOTAL.")},
            {"name": "08_skill_coverage_heatmap.html",
             "content": wrap_svg_in_html("Skill coverage per depot",
                                         skill_coverage_heatmap_svg(depots, techs),
                                         "Cells show technician headcount per (depot, skill).")},
        ]
        _ao_preview["uploaded_files"] = emit_files(files)
    except Exception as _exc:  # pragma: no cover
        logger.warning(f"emit_files failed: {_exc}")

    return {
        "objective_value": round(float(objective), 4),
        "sla_on_time_rate": round(on_time / max(1, len(tasks)), 4),
        "technician_utilization": round(util, 4),
        "total_travel_km": float(_hl.get("total_travel_km", 0.0)),
        "replenishment_alerts_count": int(_hl.get("replenishment_alerts_count", 0)),
        "solution_status": status,
        "assignments": assignments,
        "unassigned_tasks": unassigned,
        "cost_breakdown": {
            "labor_cost_eur": round(labor_total, 4),
            "travel_cost_eur": round(travel_total, 4),
            "sla_penalty_eur": round(sla_total, 4),
            "unassigned_penalty_eur": round(unassigned_pen, 4),
            "total_cost_eur": round(float(objective), 4),
        },
        "kpis": {
            "tasks_total": len(tasks),
            "tasks_assigned": len(assignments),
            "sla_on_time_rate": round(on_time / max(1, len(tasks)), 4),
            "technician_utilization": round(util, 4),
            "stockouts": 0,
        },
        "computation_metrics": {
            "wall_time_s": elapsed,
            "algorithm": "QUBO_SimulatedAnnealing",
            "iterations": n_iter,
            "num_variables": n_qubits,
            "num_qubits": n_qubits,
            "best_iter": best_iter,
            "energy_curve": energy_curve,
            "sa_temperature_schedule": {
                "T0": T0,
                "Tf": Tf,
                "n_outer": n_iter,
                "n_inner": n_inner,
            },
            "feasibility_breakdown": {
                "assignment_violation": parts["assign_pen"],
                "capacity_violation": parts["cap_pen"],
                "stock_violation": parts["stock_pen"],
            },
        },
        "additional_output": _ao_preview,
        "benchmark": {
            "execution_cost": {"value": 1.0, "unit": "credits"},
            "time_elapsed": f"{elapsed}s",
            "energy_consumption": round(0.08 * elapsed, 6),
        },
    }
