# Poller Branch Cleanup Plan

## Summary
Refactor `podterm/eventing/poller.py` to reduce nested branching by using early returns, small predicate/helper methods, and linear control flow. Preserve the existing polling behavior, log messages, queue payloads, retry timing, and public `PodPoller` interface.

## Key Changes
- Add small private helpers inside `PodPoller` for repeated branch decisions:
  - `_pod_is_terminal(pod)` identifies pods in `EXITED` or `TERMINATED` states.
  - `_parse_json_body(body)` returns decoded JSON or `None`.
  - `_check_pod_gone()` encapsulates `skip_pod_checks` plus `api_get_pod`.
- Flatten `_wait_for_running` with early returns for stop, terminal pod status, and `RUNNING`, while preserving the existing status logging cadence.
- Flatten `_wait_for_daemon` with early returns for stop, health success, auth failure, and terminal pod checks, while preserving the deadline and retry cadence.
- Flatten `_events_loop` so valid event batches, auth failures, pod-gone checks, and backoff are handled in separate linear steps.
- Flatten `_log_loop` with early continues for non-200 responses and empty bodies, while preserving offset and partial-line handling.

## Public Interfaces
No public API changes. `PodPoller`, `LogQueue`, constructor arguments, queue event shapes, emitted log text, and imported module names remain compatible.

## Test Plan
- Run `python -m pytest`.
- Add focused unit tests where practical using monkeypatches/fakes for `_request`, `api_get_pod`, `api_get_machine_logs`, and wait calls so tests avoid real sleeps and network.
- Cover the main polling exits: running, terminal pod, daemon health/auth, event cursor advancement, two-confirmation pod-gone handling, finished-run suppression, and log buffering.

## Assumptions
- The cleanup should be behavior-preserving rather than changing retry policy, timeout values, emitted messages, or event payloads.
- Helper extraction is acceptable when it directly reduces branch depth and makes the loop control flow easier to scan.
- No formatting-only sweep outside `podterm/eventing/poller.py` is needed.
