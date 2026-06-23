# pod-image — RunPod container images for GPT Caddy

Self-contained Docker build context for the three images GPT Caddy launches on RunPod.
Build everything with `./build.sh` (see below).

| Image (`ghcr.io/sysreq/…`)   | Dockerfile        | Role |
|------------------------------|-------------------|------|
| `gpt-caddy-base:latest`          | `Dockerfile.base` | Ubuntu 26.04 + baked torch/CUDA venv. Rebuilt rarely; the giant CUDA/torch `.so`s are split across layers so a cold pod pulls them in parallel. |
| `gpt-caddy-single-train:latest`  | `Dockerfile`      | `FROM` base. Bakes the rest of gpt-golf's deps, then runs `scripts/bootstrap.sh` — which clones gpt-golf at boot and runs `train_gpt.py`. GPT Caddy's default launch image. |
| `gpt-caddy-cache-server:latest`  | `Dockerfile.redis`| `FROM` base + `redis-server`. Shared torch.compile (Inductor) remote cache so cold-start compile cost is paid once per architecture, not once per run. |

## Why the dependency manifest is vendored from gpt-golf

The training pod **git-clones gpt-golf and runs its `train_gpt.py`** (the training code is *not* baked into the image). The image only bakes gpt-golf's **dependency closure** so the pod's runtime `uv sync` is a no-op. That closure is defined by gpt-golf's `pyproject.toml` + `uv.lock` (torch + flash-attn + CUDA) — **not** GPT Caddy's own root manifest (`fastapi`/`uvicorn`).

So this dir keeps a **vendored copy** of gpt-golf's `pyproject.toml` + `uv.lock`. `build.sh` refreshes them from the gpt-golf checkout (`../../gpt-golf`, override with `GPT_GOLF_DIR`) before building the train image, so the baked venv can never silently drift from gpt-golf's lock. Building from this dir as the context is what keeps the `COPY pyproject.toml uv.lock` line picking up gpt-golf's manifest instead of GPT Caddy's.

> When gpt-golf's deps change, just re-run `build.sh` — it re-vendors and rebuilds. The committed vendored copies are the fallback when no gpt-golf checkout is present.

`scripts/discover_cache.sh` is intentionally **not** here: it runs on the pod from the cloned gpt-golf repo, so it ships via git, not the image.

## Build

```bash
./build.sh                 # re-vendor deps + build base, train, cache
./build.sh --push          # ...and push to ghcr.io/sysreq
./build.sh base train      # subset (also: cache)
GPT_GOLF_DIR=/path/to/gpt-golf ./build.sh
REGISTRY=ghcr.io/you ./build.sh --push
```

Base is built first; train and cache `FROM` it. The image names are also referenced
in GPT Caddy (`config.DEFAULT_IMAGE`, `config.TEMPLATE_NAME`, and the cache-pod detector
`runpod/cli.detect_redis_server`) — keep them in sync if you rename.

### zstd layer compression on push

`--push` recompresses layers to **zstd** via [skopeo](https://github.com/containers/skopeo) (`copy --dest-compress-format zstd`, daemon → registry) instead of Docker's default gzip — smaller blobs and faster decompression mean quicker cold pulls on ephemeral pods. Requires `skopeo` (`apt install skopeo`); without it `build.sh` falls back to `docker push` (gzip). Compress harder with `ZSTD_LEVEL=19` (smaller, slower push). Uses your existing `docker login` creds.

> Compatibility: the RunPod host puller must understand zstd OCI layers (modern Docker/containerd do). If a pod ever fails to pull, push that image with plain `docker push` to fall back to gzip.
