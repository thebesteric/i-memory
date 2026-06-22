from domain.memory.models import IMemoryFiltersConfig, IMemoryGraphConfig


def test_graph_presets_recall_vs_precision():
    recall_cfg = IMemoryGraphConfig.recall_first()
    precision_cfg = IMemoryGraphConfig.precision_first()

    assert recall_cfg.enable is True
    assert precision_cfg.enable is True
    assert recall_cfg.type == "recall"
    assert precision_cfg.type == "precision"
    assert recall_cfg.max_hops >= precision_cfg.max_hops
    assert recall_cfg.per_hop_limit >= precision_cfg.per_hop_limit
    assert recall_cfg.min_relation_confidence <= precision_cfg.min_relation_confidence
    assert recall_cfg.min_walk_score <= precision_cfg.min_walk_score


def test_filters_config_merges_legacy_flat_graph_fields():
    cfg = IMemoryFiltersConfig.model_validate({
        "graph_enable": False,
        "graph_max_hops": 3,
        "graph_hop_decay": 0.85,
        "graph_per_hop_limit": 333,
        "graph_min_walk_score": 0.03,
        "graph_min_relation_confidence": 0.45,
    })

    assert cfg.graph.enable is False
    assert cfg.graph.max_hops == 3
    assert cfg.graph.hop_decay == 0.85
    assert cfg.graph.per_hop_limit == 333
    assert cfg.graph.min_walk_score == 0.03
    assert cfg.graph.min_relation_confidence == 0.45


def test_filters_config_default_graph_is_precision_first():
    cfg = IMemoryFiltersConfig()
    precision = IMemoryGraphConfig.precision_first()

    assert cfg.graph.enable is True
    assert cfg.graph.type == "precision"
    assert cfg.graph.max_hops == precision.max_hops
    assert cfg.graph.hop_decay == precision.hop_decay
    assert cfg.graph.per_hop_limit == precision.per_hop_limit
    assert cfg.graph.min_walk_score == precision.min_walk_score
    assert cfg.graph.min_relation_confidence == precision.min_relation_confidence


