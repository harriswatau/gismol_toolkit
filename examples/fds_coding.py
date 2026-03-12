
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
# 3. Real-time Fraud Detection System (FDS)
# =============================================================================

def build_fds() -> COH:
    """Construct Fraud Detection System hierarchy."""
    # TI children
    sr = COH(name="SR", attributes={"connection_status": "connected", "data_rate": 500})
    sr.methods["listen"] = lambda s: ({**s, "data_rate": s["data_rate"]+10}, 0)
    tp = COH(name="TP", attributes={"parse_success_rate": 0.99, "format": "json"})
    tp.methods["parse"] = lambda s, data: ({**s, "parse_success_rate": min(1.0, s["parse_success_rate"]+0.001)}, 0)
    ti = COH(name="TI", children=[sr, tp], attributes={"ingestion_rate": 500, "queue_size": 0, "error_rate": 0.01})

    # FE children
    ha = COH(name="HA", attributes={"aggregation_window": 60, "summary_stats": {}})
    ha.methods["aggregate"] = lambda s: (s, 0)
    bp = COH(name="BP", attributes={"user_profiles": {}, "device_profiles": {}})
    bp.methods["update_profile"] = lambda s: (s, 0)
    fe = COH(name="FE", children=[ha, bp], attributes={"features_per_sec": 500, "latency": 20, "feature_count": 50})

    # MI children
    re_rule = COH(name="RE", attributes={"rules_count": 20, "rule_hits": 0})
    re_rule.methods["evaluate_rules"] = lambda s: ({**s, "rule_hits": s["rule_hits"]+1}, 0)
    mls = COH(name="MLS", attributes={"model_name": "xgb", "version": 1, "inference_time": 15})
    mls.methods["predict"] = lambda s: (s, -s["inference_time"]/10)
    mi = COH(name="MI", children=[re_rule, mls], attributes={"inference_per_sec": 500, "latency": 15, "model_version": 1})

    # AM children
    ag = COH(name="AG", attributes={"alert_threshold": 0.8, "alert_count": 0})
    ag.methods["create_alert"] = lambda s: ({**s, "alert_count": s["alert_count"]+1}, -1)
    cm = COH(name="CM", attributes={"open_cases": 0, "assigned_analysts": 2})
    cm.methods["assign_case"] = lambda s: ({**s, "open_cases": s["open_cases"]+1}, 0)
    am = COH(name="AM", children=[ag, cm], attributes={"alert_rate": 0, "open_cases": 0, "resolution_time": 60})

    # FDS Level 1
    fds = COH(name="FDS", children=[ti, fe, mi, am], attributes={
        "tps": 500,
        "fraud_rate": 0.02,
        "false_positive_rate": 0.01,
        "alert_latency": 30
    })

    # Neural components
    fds.neural["ensemble"] = make_linear_module(10, 1)

    # Embedding
    fds.embedding = default_embedding

    # Identity constraints
    def real_time_detection(coh):
        return coh.attributes.get("alert_latency", 100) < 100
    fds.identity_constraints.append(IdentityConstraint(real_time_detection, "Real-time detection"))

    # Triggers
    def high_fraud_condition(coh):
        return coh.attributes.get("fraud_rate", 0) > 0.05
    def escalate_action(coh):
        print("  [Trigger] High fraud rate, escalating to supervisor")
    fds.trigger_constraints.append(Trigger("after_step", high_fraud_condition, escalate_action))

    def model_drift_condition(coh):
        mi_child = next((c for c in coh.children if c.name=="MI"), None)
        if mi_child and mi_child.attributes.get("model_version", 1) < 2:
            return True
        return False
    def retrain_action(coh):
        mi_child = next((c for c in coh.children if c.name=="MI"), None)
        if mi_child:
            mi_child.attributes["model_version"] = 2
            print("  [Trigger] Model drift detected, retrained to v2")
    fds.trigger_constraints.append(Trigger("after_step", model_drift_condition, retrain_action))

    # Goal constraints
    def detection_rate_goal(coh):
        return (1 - coh.attributes.get("false_positive_rate", 0)) * 100
    fds.goal_constraints.append(GoalConstraint(detection_rate_goal, weight=1.0))

    # Daemons
    class DriftDetector(Daemon):
        def run(self, coh, dt):
            # Simulate drift detection
            mi_child = next((c for c in coh.children if c.name=="MI"), None)
            if mi_child and mi_child.attributes.get("model_version", 1) == 1:
                print("  [Daemon] Checking for drift... no drift yet.")
    fds.daemons.append(DriftDetector(interval=2.0))

    return fds

def sim_fds_normal(fds):
    """Simulation 1: Normal transactions, no fraud."""
    print("\n=== FDS Simulation 1: Normal Transactions ===")
    sim = Simulator(fds, dt=1.0, max_steps=5, real_time=False)
    sim.run()
    print("Final fraud rate:", fds.attributes["fraud_rate"])

def sim_fds_fraud_detected(fds):
    """Simulation 2: Fraudulent transaction triggers alert."""
    print("\n=== FDS Simulation 2: Fraud Detected ===")
    # Simulate a fraudulent transaction
    fds.attributes["fraud_rate"] = 0.06  # above threshold
    sim = Simulator(fds, dt=1.0, max_steps=3, real_time=False)
    sim.run()
    am = next(c for c in fds.children if c.name=="AM")
    ag = next(c for c in am.children if c.name=="AG")
    print("Alerts generated:", ag.attributes["alert_count"])

def sim_fds_model_drift(fds):
    """Simulation 3: Model drift and retraining."""
    print("\n=== FDS Simulation 3: Model Drift ===")
    mi = next(c for c in fds.children if c.name=="MI")
    mi.attributes["model_version"] = 1  # old version
    sim = Simulator(fds, dt=1.0, max_steps=4, real_time=False)
    sim.run()
    print("Model version after triggers:", mi.attributes["model_version"])

