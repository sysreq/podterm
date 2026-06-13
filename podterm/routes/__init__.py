"""FastAPI route modules. Each exposes an APIRouter wired to the service singletons."""

from podterm.routes import meta, pods, runs, stream

routers = [pods.router, stream.router, meta.router, runs.router]

__all__ = ["routers", "meta", "pods", "runs", "stream"]
