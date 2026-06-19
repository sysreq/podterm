from podterm.metadata_cache import MetadataCache


def test_metadata_cache_refresh_builds_datacenters_variants_and_gpus(monkeypatch):
    cache = MetadataCache()

    monkeypatch.setattr("podterm.metadata_cache.get_datacenters", lambda: [{
        "id": "US-A",
        "gpuAvailability": [
            {"displayName": "RTX 5090", "gpuId": "NVIDIA-RTX-5090", "stockStatus": "High"},
        ],
    }])
    monkeypatch.setattr("podterm.metadata_cache.fetch_manifest", lambda: {
        "tokenizers": [{"name": "tok", "model_path": "tokenizers/tok.model"}],
        "datasets": [{
            "name": "fineweb10B_sp2048",
            "vocab_size": 2048,
            "tokenizer_name": "tok",
            "stats": {"files_train": 12},
        }],
    })

    cache.refresh()

    assert cache.ready()
    assert cache.datacenters()[0]["id"] == "US-A"
    assert cache.gpus("US-A") == [{
        "label": "RTX 5090 (High)",
        "id": "NVIDIA-RTX-5090",
    }]
    assert cache.variants()["options"] == [{
        "label": "sp2048 — vocab 2048, 12 shards",
        "id": "sp2048",
    }]


def test_metadata_cache_empty_gpu_for_unknown_datacenter():
    assert MetadataCache().gpus("missing") == []
