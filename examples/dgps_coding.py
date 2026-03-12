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
# 5. Data Governance and Privacy System (DGPS)
# =============================================================================

def build_dgps() -> COH:
    """Construct DGPS hierarchy."""
    # DC children
    mr = COH(name="MR", attributes={"metadata_entries": 2000, "last_updated": 0})
    mr.methods["store"] = lambda s, e: ({**s, "metadata_entries": s["metadata_entries"]+1}, 0)
    dlt = COH(name="DLT", attributes={"lineage_graph": {}, "nodes": 500, "edges": 1200})
    dlt.methods["track_lineage"] = lambda s: ({**s, "edges": s["edges"]+1}, 0)
    dc = COH(name="DC", children=[mr, dlt], attributes={"datasets": 100, "metadata_completeness": 0.95, "lineage_records": 500})

    # PM children
    pr = COH(name="PR", attributes={"policies": 50, "versions": 1})
    pr.methods["add_policy"] = lambda s: ({**s, "policies": s["policies"]+1}, 0)
    pep = COH(name="PEP", attributes={"enforcement_rules": 50, "decisions": 0})
    pep.methods["enforce"] = lambda s: ({**s, "decisions": s["decisions"]+1}, 0)
    pm = COH(name="PM", children=[pr, pep], attributes={"policies_count": 50, "active_rules": 45, "enforcement_points": 10})

    # AC children
    as_ac = COH(name="AS", attributes={"users": 200, "sessions": 50, "auth_methods": "MFA"})
    as_ac.methods["login"] = lambda s, success=True: ({**s, "sessions": s["sessions"]+ (1 if success else 0)}, 0)
    az = COH(name="AZ", attributes={"roles": 10, "permissions": 100, "access_matrix": {}})
    az.methods["check_permission"] = lambda s: (s, 0)
    ac = COH(name="AC", children=[as_ac, az], attributes={"users": 200, "roles": 10, "permissions": 100, "access_attempts": 5000})

    # AL children
    el = COH(name="EL", attributes={"events": 10000, "timestamps": []})
    el.methods["write_event"] = lambda s: ({**s, "events": s["events"]+1}, 0)
    cr = COH(name="CR", attributes={"reports": 5, "compliance_status": "compliant"})
    cr.methods["generate_report"] = lambda s: (s, 1)
    al = COH(name="AL", children=[el, cr], attributes={"log_volume": 10000, "retention_period": 90, "report_frequency": 30})

    # DGPS Level 1
    dgps = COH(name="DGPS", children=[dc, pm, ac, al], attributes={
        "datasets_count": 100,
        "policies_count": 50,
        "access_requests_per_sec": 10,
        "audit_events": 1000
    })

    # Neural components
    dgps.neural["anomaly_detector"] = make_linear_module(5, 1)

    # Embedding
    dgps.embedding = default_embedding

    # Identity constraints
    def gdpr_compliant(coh):
        # dummy
        return True
    dgps.identity_constraints.append(IdentityConstraint(gdpr_compliant, "GDPR compliance"))

    # Triggers
    def policy_violation_condition(coh):
        # Simulate a violation
        return False
    def alert_action(coh):
        print("  [Trigger] Policy violation, alert sent")
    dgps.trigger_constraints.append(Trigger("after_step", policy_violation_condition, alert_action))

    def data_breach_condition(coh):
        # Simulate breach
        return False
    def notify_action(coh):
        print("  [Trigger] Data breach, notification sent")
    dgps.trigger_constraints.append(Trigger("after_step", data_breach_condition, notify_action))

    # Goal constraints
    def compliance_goal(coh):
        al_child = next((c for c in coh.children if c.name=="AL"), None)
        if al_child and al_child.attributes.get("compliance_status") == "compliant":
            return 10
        return 0
    dgps.goal_constraints.append(GoalConstraint(compliance_goal, weight=1.0))

    # Daemons
    class ComplianceMonitor(Daemon):
        def run(self, coh, dt):
            al = next((c for c in coh.children if c.name=="AL"), None)
            if al:
                print(f"  [Daemon] Compliance status: {al.attributes['compliance_status']}")
    dgps.daemons.append(ComplianceMonitor(interval=2.0))

    return dgps

def sim_dgps_access(dgps):
    """Simulation 1: Access control with policy enforcement."""
    print("\n=== DGPS Simulation 1: Access Request ===")
    ac = next(c for c in dgps.children if c.name=="AC")
    as_ac = next(c for c in ac.children if c.name=="AS")
    as_ac.attributes["sessions"] = 50
    sim = Simulator(dgps, dt=1.0, max_steps=3, real_time=False)
    sim.run()
    print("Final sessions:", as_ac.attributes["sessions"])

def sim_dgps_lineage(dgps):
    """Simulation 2: Data lineage tracking."""
    print("\n=== DGPS Simulation 2: Lineage Tracking ===")
    dc = next(c for c in dgps.children if c.name=="DC")
    dlt = next(c for c in dc.children if c.name=="DLT")
    dlt.attributes["edges"] = 1200
    sim = Simulator(dgps, dt=1.0, max_steps=3, real_time=False)
    sim.run()
    print("Lineage edges after simulation:", dlt.attributes["edges"])

def sim_dgps_audit(dgps):
    """Simulation 3: Audit log and compliance reporting."""
    print("\n=== DGPS Simulation 3: Audit & Compliance ===")
    al = next(c for c in dgps.children if c.name=="AL")
    el = next(c for c in al.children if c.name=="EL")
    el.attributes["events"] = 10000
    sim = Simulator(dgps, dt=1.0, max_steps=4, real_time=False)
    sim.run()
    print("Audit events after simulation:", el.attributes["events"])

# =============================================================================
# Main runner
# =============================================================================

def main():
    print("Building all five intelligent systems...")
    bdap = build_bdap()
    re = build_re()
    fds = build_fds()
    dsps = build_dsps()
    dgps = build_dgps()

    # Run simulations for each system
    sim_bdap_normal(bdap)
    sim_bdap_load_spike(bdap)
    sim_bdap_failure(bdap)

    sim_re_normal(re)
    sim_re_new_item(re)
    sim_re_drift(re)

    sim_fds_normal(fds)
    sim_fds_fraud_detected(fds)
    sim_fds_model_drift(fds)

    sim_dsps_normal(dsps)
    sim_dsps_backpressure(dsps)
    sim_dsps_failure(dsps)

    sim_dgps_access(dgps)
    sim_dgps_lineage(dgps)
    sim_dgps_audit(dgps)

    print("\nAll simulations completed.")

if __name__ == "__main__":
    main()