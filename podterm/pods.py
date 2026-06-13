"""Pod lifecycle management — launch, auto-connect, stop, and the poller registry."""

from __future__ import annotations

import json
import secrets
import threading
import time

from pydantic import BaseModel

from podterm import db
from podterm.config import (
    DEFAULT_CLOUD_TYPE,
    DEFAULT_DATA_REPO_ID,
    DEFAULT_DATA_VARIANT,
    DEFAULT_DATA_VERSION,
    DEFAULT_IMAGE,
    POD_PREFIX,
    build_optional_debug_env,
)
from podterm.events import PodPoller
from podterm.helpers import get_local_pubkey
from podterm.pipeline import EventPipeline, pipeline
from podterm.runpod import (
    api_create_pod,
    api_terminate_pod,
    create_or_update_template,
    get_gpt_golf_pods,
    get_network_volume,
)


class LaunchConfig(BaseModel):
    branch: str
    name: str | None = None
    datacenter: str
    gpu: str
    gpu_count: int = 1
    train_script: str = "train_gpt.py"
    profile_steps: int = 0
    compile_debug: bool = False
    graph_logs: bool = False
    time_budget: int = 600
    prep_shards: int = 20
    data_repo_id: str = DEFAULT_DATA_REPO_ID
    data_version: str = DEFAULT_DATA_VERSION
    data_variant: str = DEFAULT_DATA_VARIANT
    data_path: str = ""
    tokenizer_path: str = ""
    vocab_size: str = ""
    redis_cache_server: str = ""


class PodManager:
    """Owns the per-pod PodPoller registry and the launch/connect/stop flow."""

    def __init__(self, pipeline: EventPipeline) -> None:
        self.pipeline = pipeline
        self.pollers: dict[str, PodPoller] = {}
        self.last_config: dict | None = None
        # Serialises SSH thread creation and template+pod-create to prevent races
        self._lock = threading.Lock()

    # -- queries ------------------------------------------------------------

    def list_pods(self) -> list[dict]:
        """List gg-* pods, auto-connecting event polling for any RUNNING ones."""
        try:
            pods = get_gpt_golf_pods()
        except Exception:
            pods = []
        for p in pods:
            pid = p.get("id", "")
            if p.get("desiredStatus") == "RUNNING" and pid not in self.pollers:
                self.connect(pid, p.get("name", pid), p.get("costPerHr"))
        return pods

    # -- lifecycle ----------------------------------------------------------

    def connect(self, pod_id: str, pod_name: str, cost_per_hr=None) -> None:
        """Start event polling for a pod if not already connected."""
        with self._lock:
            if pod_id in self.pollers:
                return
            run = db.get_run(pod_id)
            token = None
            if run and run.get("config_json"):
                try:
                    token = json.loads(run["config_json"]).get("eventd_token")
                except Exception:
                    token = None
            if not run:
                try:
                    db.create_run(pod_id, pod_name, {"gpu": "", "cost_per_hr": cost_per_hr})
                except Exception:
                    pass
            poller = PodPoller(pod_id, pod_name, token or "", self.pipeline.queue)
            self.pollers[pod_id] = poller  # registered even without a token so we don't retry every poll
            if token:
                poller.start()
            else:
                self.pipeline.queue.put((pod_id, "log", {"line": "No event-daemon token recorded for this pod — live stream unavailable."}))

    def launch(self, cfg: LaunchConfig) -> dict:
        """Create a template + pod, persist the run, and start its event poller."""
        cfg_dict = cfg.model_dump()
        self.last_config = dict(cfg_dict)  # copy: cfg_dict later gains eventd_token, which must not reach /api/last-config

        suffix = cfg.name or cfg.branch
        pod_name = f"{POD_PREFIX}-{suffix}-{time.strftime('%m%d-%H%M%S')}"
        token = secrets.token_urlsafe(32)

        env = {
            "EVENTD_TOKEN": token,
            "BRANCH": cfg.branch,
            "PREP_SHARDS": str(cfg.prep_shards),
            "TRAIN_SCRIPT": cfg.train_script,
            "NPROC": str(cfg.gpu_count),
            "NCCL_IB_DISABLE": "1",
            "HF_HUB_CACHE": "/workspace/.cache/huggingface",
            "DATA_REPO_ID": cfg.data_repo_id,
            "DATA_VERSION": cfg.data_version,
            "DATA_PATH": cfg.data_path or "./data/datasets/fineweb10B_sp1024",
            "DATA_VARIANT": cfg.data_variant,
            "TOKENIZER_PATH": cfg.tokenizer_path or "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": cfg.vocab_size or "1024",
            "MAX_WALLCLOCK_SECONDS": str(cfg.time_budget),
            "TRAIN_LOG_EVERY": "250",
            "VAL_LOSS_EVERY": "1000",
            "RUNPOD_ADMIN_API_KEY": "{{ RUNPOD_SECRET_SERVICE_API_KEY }}",
            "GITHUB_TOKEN": "{{ RUNPOD_SECRET_gh_gpt-golf_token }}",
            "HF_TOKEN": "{{ RUNPOD_SECRET_hf_gpt-golf_token }}",
        }
        if cfg.redis_cache_server:
            env["REDIS_CACHE_SERVER"] = cfg.redis_cache_server
        if cfg.profile_steps > 0:
            env["GPT_GOLF_PROFILE"] = str(cfg.profile_steps)
        env.update(build_optional_debug_env(cfg_dict))

        pubkey = get_local_pubkey()
        if pubkey:
            env["PUBLIC_KEY"] = pubkey

        network_vol = get_network_volume(cfg.datacenter)

        # Hold the lock across template update + pod create so concurrent
        # launches don't clobber each other's template env vars, and across
        # pollers assignment so refreshPods() can't create a duplicate.
        with self._lock:
            tpl = create_or_update_template(DEFAULT_IMAGE, env)
            pod = api_create_pod(
                pod_name, cfg.gpu, tpl,
                DEFAULT_CLOUD_TYPE, cfg.datacenter, network_vol,
                gpu_count=cfg.gpu_count,
            )
            pod_id = pod["id"]
            cost = pod.get("costPerHr", 0)
            cfg_dict["cost_per_hr"] = cost
            cfg_dict["eventd_token"] = token  # persisted in config_json so restarts can reconnect

            db.create_run(pod_id, pod_name, cfg_dict)

            poller = PodPoller(pod_id, pod_name, token, self.pipeline.queue)
            self.pollers[pod_id] = poller
            poller.start()

        return {"pod_id": pod_id, "name": pod_name, "cost_per_hr": cost}

    def stop(self, pod_id: str) -> dict:
        """Terminate a pod, stop its poller, and finalize the run row."""
        api_terminate_pod(pod_id)
        with self._lock:
            poller = self.pollers.pop(pod_id, None)
        if poller:
            poller.stop()
        self.pipeline.finalize_run(pod_id)
        return {"status": "terminated", "pod_id": pod_id}

    def stop_all(self) -> None:
        """Stop every active poller (used on app shutdown)."""
        for p in self.pollers.values():
            p.stop()


manager = PodManager(pipeline)
