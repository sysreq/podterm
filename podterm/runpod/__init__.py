"""RunPod API layer.

All RunPod interaction goes through the `runpodctl` CLI (cli.py), with two
sanctioned read-only exceptions confined to api.py: a GraphQL query for
utilization telemetry and the hapi machine-log endpoint for boot/image-pull
progress (the latter authenticated via console.py's Clerk JWT minting).

This package re-exports the public surface so callers keep using
`from podterm.runpod import X`.
"""

from podterm.runpod.api import api_get_machine_logs, api_get_telemetry
from podterm.runpod.cli import (
    TEMPLATE_PORTS,
    api_create_pod,
    api_get_pod,
    api_get_ssh_info,
    api_list_pods,
    api_terminate_pod,
    create_or_update_template,
    detect_redis_server,
    find_template,
    get_available_gpus,
    get_datacenters,
    get_gpt_golf_pods,
    get_network_volume,
)

__all__ = [
    "TEMPLATE_PORTS",
    "api_create_pod",
    "api_get_machine_logs",
    "api_get_pod",
    "api_get_ssh_info",
    "api_get_telemetry",
    "api_list_pods",
    "api_terminate_pod",
    "create_or_update_template",
    "detect_redis_server",
    "find_template",
    "get_available_gpus",
    "get_datacenters",
    "get_gpt_golf_pods",
    "get_network_volume",
]
