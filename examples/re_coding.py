
# =============================================================================
# 2. Recommendation Engine (RE)
# =============================================================================

def build_re() -> COH:
    """Construct Recommendation Engine hierarchy."""
    # UPM children
    udb = COH(name="UDB", attributes={"user_records": 1000, "access_latency": 5})
    udb.methods["query"] = lambda s: (s, -s["access_latency"])
    ubt = COH(name="UBT", attributes={"events_per_sec": 50, "session_duration": 300})
    ubt.methods["track_event"] = lambda s: ({**s, "events_per_sec": s["events_per_sec"]+1}, 0)
    upm = COH(name="UPM", children=[udb, ubt], attributes={"user_count": 1000, "active_users": 200})

    # ICM children
    idb = COH(name="IDB", attributes={"item_records": 5000, "access_latency": 3})
    idb.methods["query"] = lambda s: (s, -s["access_latency"])
    ife = COH(name="IFE", attributes={"features_extracted": 5000, "processing_time": 10})
    ife.methods["extract_features"] = lambda s: ({**s, "features_extracted": s["features_extracted"]+1}, 0)
    icm = COH(name="ICM", children=[idb, ife], attributes={"item_count": 5000, "new_items_per_day": 100})

    # RG children
    cfe = COH(name="CFE", attributes={"user_similarity": "matrix", "update_frequency": 3600})
    cfe.methods["predict_rating"] = lambda s: (s, 0)
    cbe = COH(name="CBE", attributes={"item_profiles": "content", "similarity_threshold": 0.7})
    cbe.methods["compute_content_scores"] = lambda s: (s, 0)
    hs = COH(name="HS", attributes={"combination_weights": [0.5, 0.5], "current_strategy": "hybrid"})
    hs.methods["combine_scores"] = lambda s: (s, 0)
    rg = COH(name="RG", children=[cfe, cbe, hs], attributes={
        "requests_per_sec": 100, "avg_latency": 50, "diversity_score": 0.8
    })

    # FC children
    csa = COH(name="CSA", attributes={"click_rate": 10, "patterns": {}})
    csa.methods["process_click"] = lambda s: ({**s, "click_rate": s["click_rate"]+1}, 1)
    rp = COH(name="RP", attributes={"rating_distribution": {}, "average_rating": 4.2})
    rp.methods["process_rating"] = lambda s, r=5: ({**s, "average_rating": (s["average_rating"]+r)/2}, 0)
    fc = COH(name="FC", children=[csa, rp], attributes={"feedback_rate": 50, "clicks_per_sec": 10, "ratings_per_sec": 5})

    # RE Level 1
    re = COH(name="RE", children=[upm, icm, rg, fc], attributes={
        "total_users": 1000,
        "total_items": 5000,
        "requests_per_sec": 100,
        "avg_response_time": 50,
        "click_through_rate": 0.05
    })

    # Neural components
    re.neural["ensemble_model"] = make_linear_module(10, 1)  # dummy

    # Embedding
    re.embedding = default_embedding

    # Identity constraints
    def personalized_constraint(coh):
        # Dummy
        return True
    re.identity_constraints.append(IdentityConstraint(personalized_constraint, "Personalized"))

    # Triggers
    def new_interaction_condition(coh):
        fc_child = next((c for c in coh.children if c.name=="FC"), None)
        if fc_child and fc_child.attributes.get("feedback_rate", 0) > 60:
            return True
        return False
    def update_profile_action(coh):
        upm_child = next((c for c in coh.children if c.name=="UPM"), None)
        if upm_child:
            upm_child.attributes["active_users"] += 10
            print("  [Trigger] New interaction, updating profile")
    re.trigger_constraints.append(Trigger("after_step", new_interaction_condition, update_profile_action))

    def ctr_drop_condition(coh):
        return coh.attributes.get("click_through_rate", 0.05) < 0.03
    def retrain_action(coh):
        print("  [Trigger] CTR dropped, retraining models")
        # Simulate retraining
    re.trigger_constraints.append(Trigger("after_step", ctr_drop_condition, retrain_action))

    # Goal constraints
    def ctr_goal(coh):
        return coh.attributes.get("click_through_rate", 0) * 100
    re.goal_constraints.append(GoalConstraint(ctr_goal, weight=1.0))

    def diversity_goal(coh):
        rg_child = next((c for c in coh.children if c.name=="RG"), None)
        if rg_child:
            return rg_child.attributes.get("diversity_score", 0) * 10
        return 0
    re.goal_constraints.append(GoalConstraint(diversity_goal, weight=0.5))

    # Daemons
    class PerformanceMonitorRE(Daemon):
        def run(self, coh, dt):
            print(f"  [Daemon] RE performance: CTR={coh.attributes['click_through_rate']:.3f}")
    re.daemons.append(PerformanceMonitorRE(interval=1.0))

    return re

def sim_re_normal(re):
    """Simulation 1: Normal recommendation flow."""
    print("\n=== RE Simulation 1: Normal Operation ===")
    sim = Simulator(re, dt=1.0, max_steps=5, real_time=False)
    sim.run()
    print("Final CTR:", re.attributes["click_through_rate"])

def sim_re_new_item(re):
    """Simulation 2: New item added, catalog updated."""
    print("\n=== RE Simulation 2: New Item Addition ===")
    icm = next(c for c in re.children if c.name=="ICM")
    icm.attributes["item_count"] += 1
    # Trigger catalog update via event
    re.publish("new_item", {"item_id": 5001})
    sim = Simulator(re, dt=1.0, max_steps=3, real_time=False)
    sim.run()
    print("Item count now:", icm.attributes["item_count"])

def sim_re_drift(re):
    """Simulation 3: Model drift and retraining."""
    print("\n=== RE Simulation 3: Model Drift ===")
    # Simulate CTR drop
    re.attributes["click_through_rate"] = 0.02
    sim = Simulator(re, dt=1.0, max_steps=3, real_time=False)
    sim.run()
    print("CTR after triggers:", re.attributes["click_through_rate"])
