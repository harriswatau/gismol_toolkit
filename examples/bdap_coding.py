#!/usr/bin/env python3
"""
Five Intelligent Systems modelled with GISMOL (COH framework)
Each system is built as a hierarchy of COH objects with attributes, methods,
neural components, constraints, triggers, and daemons.
Three simulations per system demonstrate key behaviors.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Callable
import time

# GISMOL imports (assumes package installed)
from gismol.core import COH, NeuralModule, Trigger, Daemon, ConstraintViolation
from gismol.constraints import IdentityConstraint, GoalConstraint
from gismol.simulation import Simulator, Event, EventBus
from gismol.utils import default_embedding, to_json, from_json
from gismol.visualization import draw_hierarchy  # optional

# =============================================================================
# Helper functions and dummy neural modules
# =============================================================================
import torch
import torch.nn as nn
import random
from gismol.core import COH, NeuralModule

def make_linear_module(in_features: int, out_features: int) -> NeuralModule:
    """
    Create a simple linear neural module wrapped as a GISMOL NeuralModule.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
    
    Returns:
        NeuralModule: A GISMOL neural component containing a linear layer
                      and an Adam optimizer.
    """
    net = nn.Linear(in_features, out_features)
    # Wrap the network with an Adam optimizer (learning rate 0.01)
    return NeuralModule(net, optimizer_class=torch.optim.Adam, lr=0.01)


def default_policy(coh: COH) -> str:
    """
    A random policy for simulation steps. Selects a random method name
    from the COH object's methods dictionary. If there are no methods,
    returns an empty string.
    
    Args:
        coh (COH): The COH object whose methods are to be chosen from.
    
    Returns:
        str: The name of a randomly selected method, or an empty string.
    """
    if coh.methods:
        return random.choice(list(coh.methods.keys()))
    return ""
# =============================================================================
# 1. Big Data Analytics Platform (BDAP)
# =============================================================================

def build_bdap() -> COH:
    """Construct the BDAP hierarchy."""
    # ---------- Level 3 components ----------
    # DIM children
    sc = COH(name="SC", attributes={
        "connection_status": "connected",
        "data_rate": 100,
        "source_type": "sensor",
        "last_poll_time": 0
    })
    sc.methods["poll_data"] = lambda s, rate=10: (
        {**s, "data_rate": s["data_rate"] + rate, "last_poll_time": s["last_poll_time"]+1},
        0
    )
    sc.methods["connect"] = lambda s: ({**s, "connection_status": "connected"}, 0)
    sc.methods["disconnect"] = lambda s: ({**s, "connection_status": "disconnected"}, 0)

    mq = COH(name="MQ", attributes={
        "queue_depth": 0,
        "message_count": 0,
        "consumer_count": 1,
        "throughput": 0
    })
    mq.methods["produce"] = lambda s, n=1: (
        {**s, "queue_depth": s["queue_depth"] + n, "message_count": s["message_count"] + n},
        0
    )
    mq.methods["consume"] = lambda s, n=1: (
        {**s, "queue_depth": max(0, s["queue_depth"] - n)},
        0
    )

    dpp = COH(name="DPP", attributes={
        "preprocessing_steps": 3,
        "data_quality_metrics": 1.0,
        "error_rate": 0.0
    })
    dpp.methods["clean"] = lambda s: ({**s, "data_quality_metrics": min(1.0, s["data_quality_metrics"]+0.1)}, 1)
    dpp.methods["transform"] = lambda s: (s, 0)

    dim = COH(name="DIM", children=[sc, mq, dpp], attributes={
        "ingestion_rate": 100,
        "queue_size": 0,
        "active_connections": 1,
        "error_count": 0
    })
    dim.methods["start_ingestion"] = lambda s: (s, 0)

    # DSM children
    dfs = COH(name="DFS", attributes={
        "capacity": 10000,
        "used": 2000,
        "replication": 3,
        "health": "healthy"
    })
    dfs.methods["write"] = lambda s, size=100: (
        {**s, "used": s["used"] + size},
        0
    )

    ndb = COH(name="NDB", attributes={
        "records": 5000,
        "query_latency": 10,
        "write_latency": 5
    })
    ndb.methods["query"] = lambda s: (s, -s["query_latency"]/10)

    mds = COH(name="MDS", attributes={
        "metadata_entries": 1000,
        "consistency": True
    })
    mds.methods["store_metadata"] = lambda s: (s, 0)

    dsm = COH(name="DSM", children=[dfs, ndb, mds], attributes={
        "storage_used": 2000,
        "available_space": 8000,
        "read_latency": 5,
        "write_latency": 8,
        "replication_factor": 3
    })

    # DPM children
    bpe = COH(name="BPE", attributes={
        "jobs_pending": 2,
        "jobs_running": 1,
        "completion_time": 10
    })
    bpe.methods["submit_batch_job"] = lambda s: ({**s, "jobs_pending": s["jobs_pending"]+1}, -1)

    spe = COH(name="SPE", attributes={
        "events_per_second": 1000,
        "latency": 50,
        "throughput": 900
    })
    spe.methods["process_event"] = lambda s: ({**s, "latency": s["latency"]-1}, 1)

    mle = COH(name="MLE", attributes={
        "models_deployed": 2,
        "training_jobs": 0,
        "inference_latency": 20
    })
    mle.methods["predict"] = lambda s: (s, -s["inference_latency"]/10)

    dpm = COH(name="DPM", children=[bpe, spe, mle], attributes={
        "jobs_running": 3,
        "queue_length": 5,
        "cpu_usage": 60,
        "memory_usage": 50,
        "job_success_rate": 0.95
    })

    # DVM children
    db = COH(name="DB", attributes={
        "widgets": 5,
        "refresh_rate": 10,
        "user_interactions": 100
    })
    db.methods["render"] = lambda s: ({**s, "refresh_rate": s["refresh_rate"]+1}, 0)

    rt = COH(name="RT", attributes={
        "scheduled_reports": 3,
        "delivery_status": "success"
    })
    rt.methods["generate_report"] = lambda s: (s, 1)

    as_ = COH(name="AS", attributes={
        "alert_rules": 5,
        "triggered_alerts": 0
    })
    as_.methods["send_alert"] = lambda s: ({**s, "triggered_alerts": s["triggered_alerts"]+1}, 0)

    dvm = COH(name="DVM", children=[db, rt, as_], attributes={
        "active_dashboards": 2,
        "report_generation_rate": 1,
        "alert_count": 0,
        "user_sessions": 10
    })

    # ---------- Level 1 BDAP ----------
    bdap = COH(name="BDAP", children=[dim, dsm, dpm, dvm], attributes={
        "overall_status": "running",
        "total_data_volume": 10000,
        "processing_load": 50,
        "user_count": 100,
        "config_version": "1.0"
    })

    # Neural components
    # Workload predictor (LSTM) – simplified to a linear module
    bdap.neural["workload_predictor"] = make_linear_module(5, 1)

    # Embedding (default flattens numerical attributes)
    bdap.embedding = default_embedding

    # Identity constraints
    def at_least_one_source(coh):
        dim_child = next((c for c in coh.children if c.name=="DIM"), None)
        if dim_child:
            sc_child = next((c for c in dim_child.children if c.name=="SC"), None)
            return sc_child is not None
        return False
    bdap.identity_constraints.append(IdentityConstraint(at_least_one_source, "At least one data source"))

    def components_compatible(coh):
        # Dummy: always true in simulation
        return True
    bdap.identity_constraints.append(IdentityConstraint(components_compatible, "Components compatible"))

    # Trigger constraints
    def high_load_condition(coh):
        return coh.attributes.get("processing_load", 0) > 80
    def scale_action(coh):
        coh.attributes["processing_load"] = max(50, coh.attributes["processing_load"] - 20)
        print("  [Trigger] High load detected, scaling up")
    bdap.trigger_constraints.append(Trigger("after_step", high_load_condition, scale_action))

    def component_failure_condition(coh):
        # Simulate random failure (here we just check a dummy flag)
        return False  # not activated by default
    def alert_action(coh):
        print("  [Trigger] Component failure, alert sent")
    bdap.trigger_constraints.append(Trigger("after_step", component_failure_condition, alert_action))

    # Goal constraints (minimize latency, maximize throughput)
    def latency_goal(coh):
        dpm_child = next((c for c in coh.children if c.name=="DPM"), None)
        if dpm_child and "queue_length" in dpm_child.attributes:
            return -dpm_child.attributes["queue_length"]  # negative because we want to minimize
        return 0
    bdap.goal_constraints.append(GoalConstraint(latency_goal, weight=1.0))

    def throughput_goal(coh):
        dim_child = next((c for c in coh.children if c.name=="DIM"), None)
        if dim_child:
            return dim_child.attributes.get("ingestion_rate", 0)
        return 0
    bdap.goal_constraints.append(GoalConstraint(throughput_goal, weight=0.01))

    # Daemons
    class HealthMonitor(Daemon):
        def run(self, coh, dt):
            print(f"  [Daemon] Health check: status={coh.attributes['overall_status']}")
    bdap.daemons.append(HealthMonitor(interval=2.0))

    class PerformanceMonitor(Daemon):
        def run(self, coh, dt):
            load = coh.attributes.get("processing_load", 0)
            print(f"  [Daemon] Performance: load={load}%")
    bdap.daemons.append(PerformanceMonitor(interval=1.0))

    return bdap

# Simulations for BDAP
def sim_bdap_normal(bdap):
    """Simulation 1: Normal operation."""
    print("\n=== BDAP Simulation 1: Normal Operation ===")
    sim = Simulator(bdap, dt=1.0, max_steps=5, real_time=False)
    sim.run(policy=lambda c: "poll_data" if c.name=="SC" else None)  # simplistic policy
    print("Simulation finished.")

def sim_bdap_load_spike(bdap):
    """Simulation 2: Load spike triggers scaling."""
    print("\n=== BDAP Simulation 2: Load Spike ===")
    # Manually increase load
    bdap.attributes["processing_load"] = 90
    sim = Simulator(bdap, dt=1.0, max_steps=5, real_time=False)
    # Override condition to trigger scaling
    sim.run()
    print("Final load:", bdap.attributes["processing_load"])

def sim_bdap_failure(bdap):
    """Simulation 3: Component failure and recovery."""
    print("\n=== BDAP Simulation 3: Component Failure ===")
    # Simulate a failure in DPM
    dpm = next(c for c in bdap.children if c.name=="DPM")
    dpm.attributes["jobs_running"] = 0
    # Add a trigger that reacts to failure (we'll simulate via event)
    def failure_condition(coh):
        return coh.attributes.get("jobs_running", 1) == 0
    def recover_action(coh):
        coh.attributes["jobs_running"] = 3
        print("  [Trigger] Recovery action: restarted DPM")
    # Temporarily add trigger
    dpm.trigger_constraints.append(Trigger("after_step", failure_condition, recover_action))
    sim = Simulator(bdap, dt=1.0, max_steps=3, real_time=False)
    sim.run()
    print("Final jobs_running:", dpm.attributes["jobs_running"])
