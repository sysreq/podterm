import argparse
import sys
import traceback


def main() -> int:
    p = argparse.ArgumentParser(prog='python -m podterm.diagnostics',
                                description='Run diagnostics on a saved checkpoint. Set DEVICE=cpu to run off-GPU; '
                                            '--no-tokens skips the forward/token stages (weights-only, no dataset needed).')
    p.add_argument('checkpoint'); p.add_argument('--out', default=None)
    p.add_argument('--no-tokens', action='store_true', help='skip forward/token stages (weights-only)')
    p.add_argument('--step', type=int, default=None, help='training step this checkpoint came from (recorded in meta)')
    p.add_argument('--request-id', default=None,
                   help='opaque id echoed into meta.request_id so the caller can reject a stale doc')
    a = p.parse_args()

    import torch, train_gpt  # noqa: E402 -- import runs config_gpt.settings(): device (DEVICE env), tokenizer, seeds

    # The model is uncompiled here, so its eager flex_attention is the slow reference path (minutes/forward).
    # Swap in the SDPA equivalent -- same masked attention, native fused kernel, no compile -- before any forward.
    from podterm.diagnostics.attention import sdpa_flex  # noqa: E402
    train_gpt.flex_attention = sdpa_flex

    model = train_gpt.GPT().to(train_gpt.HW.device).bfloat16()
    # weights_only=True: the checkpoint is an untrusted state_dict pulled off a remote pod, so refuse
    # to unpickle arbitrary objects while loading it.
    state = torch.load(a.checkpoint, map_location=train_gpt.HW.device, weights_only=True)
    model.load_state_dict(state, strict=True)
    from podterm.diagnostics.runner import run_diagnostics  # noqa: E402

    val_tokens = None if a.no_tokens else train_gpt.load_validation_tokens()
    out = a.out or a.checkpoint.rsplit('.', 1)[0] + '.diag.json'
    meta = dict(checkpoint=a.checkpoint)
    if a.step is not None: meta['step'] = a.step
    if a.request_id is not None: meta['request_id'] = a.request_id
    # run_diagnostics re-raises on a fatal error (it writes the doc atomically on success only).
    run_diagnostics(model, None, val_tokens, json_path=out, extra_meta=meta)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 -- any fatal error must become a non-zero exit, not a stale doc
        traceback.print_exc()
        sys.exit(1)
