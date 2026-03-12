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
# 4. Distributed Stream Processing System (DSPS)
# =============================================================================

def build_dsps() -> COH:
    """Construct DSPS hierarchy."""
    # SI children
    sc_si = COH(name="SC", attributes={"connection": "up", "data_rate": 1000})
    sc_si.methods["poll"] = lambda s: ({**s, "data_rate": s["data_rate"]+50}, 0)
    pt = COH(name="PT", attributes={"partition_key": "hash", "distribution": "balanced"})
    pt.methods["partition"] = lambda s: (s, 0)
    si = COH(name="SI", children=[sc_si, pt], attributes={"input_rate": 1000, "partition_count": 4, "backlog": 0})

    # SP children
    op = COH(name="OP", attributes={"function": "filter", "parallelism": 2, "backlog": 0})
    op.methods["execute"] = lambda s: ({**s, "backlog": max(0, s["backlog"]-10)}, 1)
    wd = COH(name="WD", attributes={"window_type": "tumbling", "size": 10, "slide": 10})
    wd.methods["add_event"] = lambda s: (s, 0)
    sp = COH(name="SP", children=[op, wd], attributes={"processing_rate": 800, "operator_count": 2, "watermark": 100})

    # SM children
    kvs = COH(name="KVS", attributes={"entries": 5000, "read_latency": 2, "write_latency": 3})
    kvs.methods["get"] = lambda s, key: (s, -s["read_latency"])
    ck = COH(name="CK", attributes={"checkpoint_interval": 60, "last_checkpoint": 0})
    ck.methods["save"] = lambda s: ({**s, "last_checkpoint": s["last_checkpoint"]+1}, 0)
    sm = COH(name="SM", children=[kvs, ck], attributes={"state_size": 10000, "access_latency": 5, "checkpoint_frequency": 60})

    # OS children
    sk = COH(name="SK", attributes={"destination": "kafka", "write_rate": 800})
    sk.methods["write"] = lambda s: ({**s, "write_rate": s["write_rate"]+10}, 0)
    of = COH(name="OF", attributes={"format": "avro", "schema": "v1"})
    of.methods["format"] = lambda s: (s, 0)
    os = COH(name="OS", children=[sk, of], attributes={"output_rate": 800, "sink_latency": 10, "error_rate": 0})

    # DSPS Level 1
    dsps = COH(name="DSPS", children=[si, sp, sm, os], attributes={
        "events_per_sec": 1000,
        "processing_latency": 50,
        "throughput": 900,
        "checkpoint_status": "ok"
    })

    # Neural components
    dsps.neural["auto_scaler"] = make_linear_module(3, 1)

    # Embedding
    dsps.embedding = default_embedding

    # Identity constraints
    def exactly_once(coh):
        return True  # placeholder
    dsps.identity_constraints.append(IdentityConstraint(exactly_once, "Exactly-once processing"))

    # Triggers
    def backpressure_condition(coh):
        si_child = next((c for c in coh.children if c.name=="SI"), None)
        return si_child and si_child.attributes.get("backlog", 0) > 100
    def scale_action(coh):
        sp_child = next((c for c in coh.children if c.name=="SP"), None)
        if sp_child:
            sp_child.attributes["operator_count"] += 1
            print("  [Trigger] Backpressure, scaled out operators")
    dsps.trigger_constraints.append(Trigger("after_step", backpressure_condition, scale_action))

    def node_failure_condition(coh):
        # Simulate random failure
        return False
    def restart_action(coh):
        print("  [Trigger] Node failure, restarting")
    dsps.trigger_constraints.append(Trigger("after_step", node_failure_condition, restart_action))

    # Goal constraints
    def latency_goal(coh):
        return -coh.attributes.get("processing_latency", 100)
    dsps.goal_constraints.append(GoalConstraint(latency_goal, weight=1.0))

    # Daemons
    class LagMonitor(Daemon):
        def run(self, coh, dt):
            si = next((c for c in coh.children if c.name=="SI"), None)
            if si:
                print(f"  [Daemon] Backlog: {si.attributes['backlog']}")
    dsps.daemons.append(LagMonitor(interval=1.0))

    return dsps

def sim_dsps_normal(dsps):
    """Simulation 1: Normal stream processing."""
    print("\n=== DSPS Simulation 1: Normal Operation ===")
    sim = Simulator(dsps, dt=1.0, max_steps=5, real_time=False)
    sim.run()
    print("Final processing latency:", dsps.attributes["processing_latency"])

def sim_dsps_backpressure(dsps):
    """Simulation 2: Backpressure triggers scaling."""
    print("\n=== DSPS Simulation 2: Backpressure ===")
    si = next(c for c in dsps.children if c.name=="SI")
    si.attributes["backlog"] = 150  # cause trigger
    sim = Simulator(dsps, dt=1.0, max_steps=3, real_time=False)
    sim.run()
    sp = next(c for c in dsps.children if c.name=="SP")
    print("Operator count after scaling:", sp.attributes["operator_count"])

def sim_dsps_failure(dsps):
    """Simulation 3: Node failure and recovery from checkpoint."""
    print("\n=== DSPS Simulation 3: Failure Recovery ===")
    sm = next(c for c in dsps.children if c.name=="SM")
    ck = next(c for c in sm.children if c.name=="CK")
    # Simulate failure: set checkpoint old
    ck.attributes["last_checkpoint"] = 0
    # Manually trigger restore via daemon? We'll simulate a failure event.
    dsps.publish("node_failure", {"node": "sp"})
    sim = Simulator(dsps, dt=1.0, max_steps=3, real_time=False)
    sim.run()
    print("Checkpoint after recovery:", ck.attributes["last_checkpoint"])
