from pathlib import Path


def test_removed_log_workflow_has_no_static_references():
    text = "\n".join(p.read_text() for p in Path("podterm/static").rglob("*") if p.is_file())
    for needle in (
        "/api/logs",
        "/logs/import",
        "/api/runs/${podId}/log",
        "btn-import-logs",
        "import-dialog",
        "openImportDialog",
    ):
        assert needle not in text
