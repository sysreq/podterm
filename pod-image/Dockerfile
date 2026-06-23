FROM ghcr.io/sysreq/gpt-caddy-base:latest

ENV REPO_OWNER=sysreq \
    REPO_NAME=gpt-golf \
    BRANCH=main \
    TRAIN_SCRIPT=train_gpt.py \
    PREP_SHARDS=20 \
    DATA_REPO_ID=sysrekt/parameter-golf \
    DATA_VERSION=main \
    HF_HUB_CACHE=/workspace/.cache/huggingface \
    DATA_VARIANT=sp1024

# Sync remaining deps (small python packages + non-torch project deps) into the
# baked venv. torch + CUDA libs are already installed in the base image, so uv
# sync only adds the lightweight project dependencies.
# NB: these are gpt-golf's VENDORED pyproject.toml + uv.lock (this dir), not
# PodTerm's own — the pod clones gpt-golf and runs train_gpt.py, so the baked
# venv must satisfy gpt-golf's closure for the pod's runtime `uv sync` to no-op.
# build.sh refreshes them from the gpt-golf checkout before building.
COPY pyproject.toml uv.lock /opt/project/
RUN cd /opt/project && \
    UV_PROJECT_ENVIRONMENT=/opt/.venv uv sync --no-dev --no-install-project

# Bootstrap script + event daemon
COPY scripts/bootstrap.sh /usr/local/bin/bootstrap.sh
COPY scripts/pod_eventd.py /usr/local/bin/pod_eventd.py
RUN chmod +x /usr/local/bin/bootstrap.sh

CMD ["/usr/local/bin/bootstrap.sh"]
