#!/usr/bin/env bash
# Replace never-called CUDA libs with tiny symbol-exact stubs.
#
# torch DT_NEEDEDs libcufft/libcusparseLt, so the loader must map them (and resolve their
# versioned symbol nodes) at `import torch` even though this GPT issues no FFT or 2:4-
# structured-sparse matmul -- a DT_NEEDED lib must be present at load even under lazy binding,
# so deleting it ImportErrors and an empty stub fails the version check. A 16 KB no-op stub
# exporting exactly the symbols torch references, at their version node, satisfies the loader;
# the stubbed functions are never called, so ~500 MB of unused kernels never ship. Symbols are
# auto-derived from the installed torch libs, so a torch bump self-corrects (re-run; no drift).
#
# cusparse is deliberately left intact (sparse-gradient paths can reach it). Re-add the real
# wheels (drop this step) if the model ever calls torch.fft.* or a structured-sparse matmul --
# a stubbed entry point would no-op and corrupt results.
#
# libnccl.so.2 is DT_NEEDED but only exercised multi-GPU (ProcessGroupNCCL); stubbed here so
# single-GPU pulls (the 99% case) skip ~206 MB compressed. bootstrap.sh reinstalls the real wheel before
# torchrun when NPROC>1 -- a stubbed collective would corrupt a multi-GPU run, so that restore
# is fail-loud.
#
# Usage: stub_unused_cuda_libs.sh <site-packages-dir>   (run after torch is installed)
set -euo pipefail
SP="$1"
# Fail loud if the caller's python*/site-packages glob didn't expand (e.g. a Python bump):
# without this the script would silently no-op and ship the full ~500 MB.
[ -d "$SP/torch/lib" ] || { echo "stub: '$SP' is not a site-packages with torch installed" >&2; exit 1; }
LIBDIRS=("$SP"/nvidia/cu13/lib "$SP"/nvidia/cusparselt/lib "$SP"/nvidia/nccl/lib)

# bare names of every symbol torch leaves undefined (i.e. expects a CUDA lib to provide)
UNDEF="$(for f in "$SP"/torch/lib/*.so; do nm -D --undefined-only "$f" 2>/dev/null; done \
         | awk '{print $NF}' | sed 's/@.*//' | sort -u)"

find_lib() { for d in "${LIBDIRS[@]}"; do [ -e "$d/$1" ] && { echo "$d/$1"; return; }; done; }
drop()     { local p="$1"; [ -n "$p" ] && rm -f "$(readlink -f "$p")" "$p"; }  # symlink + uv-cache target

stub() {
    local soname="$1" real; real="$(find_lib "$soname")"
    [ -n "$real" ] || { echo "stub: $soname not found, skipping"; return; }
    local syms vernode="" d bare
    : >/tmp/_syms
    for d in $(nm -D --defined-only "$real" 2>/dev/null | awk '{print $NF}'); do
        bare="${d%@@*}"; bare="${bare%@*}"
        printf '%s\n' "$UNDEF" | grep -qx "$bare" || continue
        echo "$bare" >>/tmp/_syms
        case "$d" in *@@*) vernode="${d##*@@}";; esac
    done
    sort -u /tmp/_syms -o /tmp/_syms
    [ -s /tmp/_syms ] || { echo "stub: $soname has no torch-referenced symbols, skipping"; return; }
    awk '{print "void "$1"(void){}"}' /tmp/_syms >/tmp/_stub.c
    local vs=""
    if [ -n "$vernode" ]; then
        { echo "$vernode {"; echo " global:"; awk '{print "  "$1";"}' /tmp/_syms; echo " local: *;"; echo "};"; } >/tmp/_ver.map
        vs="-Wl,--version-script=/tmp/_ver.map"
    fi
    gcc -shared -fPIC $vs -Wl,-soname,"$soname" -o /tmp/_stub.so /tmp/_stub.c
    drop "$real"; cp /tmp/_stub.so "$real"
    echo "stubbed $soname ($(wc -l </tmp/_syms) syms, version='${vernode:-none}')"
}

stub libcufft.so.12
stub libcusparseLt.so.0
stub libnccl.so.2                     # restored at runtime by bootstrap.sh when NPROC>1
drop "$(find_lib libcufftw.so.12)"   # orphan: needed by nothing
