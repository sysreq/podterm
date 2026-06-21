# Screen Scaling Implementation Plan



## Goal

Make the KPI metric boxes and dialogs scale to the actual presentation size instead of assuming a near-static desktop width/height.

The dashboard should remain readable and stable across:

- Narrow laptop windows
- Wide desktop displays
- Browser zoom / presentation mode
- Smaller viewport heights
- Hidden/revealed tab containers where Plotly needs resize handling


## Current State

Relevant files:

- `podterm/static/css/tokens.css`
  - Fixed dashboard height contract:
    - `--h-kpi: 108px`
    - `--h-banner: 56px`
    - `--h-baseline: 44px`
    - `--h-chart: 300px`
    - `--h-diag: 136px`
    - `--h-bottom: 292px`
  - Only one typography breakpoint at `max-width: 1600px`.
- `podterm/static/css/layout.css`
  - Main shell uses fixed sidebar width and fixed dashboard section heights.
  - `#kpi-row` is always `repeat(9, minmax(0, 1fr))`.
  - `#diag-row` is always `repeat(5, minmax(0, 1fr))`.
- `podterm/static/css/components.css`
  - KPI values/subtext are mostly single-line with ellipsis.
  - Narrow viewport support only shrinks text at one breakpoint.
  - Race and baseline rows use fixed/fractional widths that can crowd.
- `podterm/static/css/base.css`
  - All `<dialog>` elements use fixed `width: 520px`.
- `podterm/static/js/cards.js`
  - KPI cards have stable internal slots, but no size-aware behavior.
- `podterm/static/js/app.js`
  - Plotly charts resize only on tab reveal, not on layout/container changes.


## Validation Corrections (resolved — apply these)

Cross-checked against the code; these override the rougher assumptions below.

1. Real card counts: the grids over-declare tracks.
   - `#kpi-row` declares `repeat(9, …)` but only **6** KPI cards render
     (`kpis.js` builds: projected, hero, loss, gpu, cost, health). Three tracks are empty today.
   - `#diag-row` declares `repeat(5, …)` but only **2** diag cards render (vbpb, system).
   - Size `auto-fit` `minmax()` mins against the real 6 / 2 counts, not 9 / 5. Decision: fit real counts.

2. Diagnostics selector disambiguation.
   - The cards inside `#diag-row` use the class `.kpi-card.diag` (a KPI-card variant), styled at
     `components.css:99-100`. Adapt **those**.
   - Do **not** confuse them with `.diag-card` / `.diag-cards` (`components.css:160-169`), which is the
     unrelated Model Health panel grid that already uses `repeat(auto-fill, minmax(180px, 1fr))`.

3. Chart-height floor is mandatory (biggest trap).
   - Plotly chart height is **JS-driven**, not CSS-flex: `charts.js` passes `height: lossEl.clientHeight || 280`
     into the layout at init; `compare.js` hardcodes `height: 300`. `Plotly.Plots.resize()` re-reads the box.
   - The live chain is `#charts-row{height:var(--h-chart)} → .chart-box(flex col) → .chart-plot{flex:1;min-height:0}`.
     If `--h-chart` is loosened to `auto`/pure flex with no floor, `.chart-plot` collapses to ~0 and Plotly
     renders a zero-height chart. **Always keep a concrete `min-height` on `#charts-row`/`.chart-box`** — the
     resize observer fixes width reflow, not this.

4. `--fs-kpi` coupling.
   - `.kpi-value` uses `--fs-kpi` for both `font-size` **and** `min-height: calc(var(--fs-kpi)*1.2)`
     (`components.css:71-73`). Switching the font-size to `clamp()` without also revising that `min-height`
     desyncs the value-slot height (a no-shift regression). Update both together. `--fs-hero` rides alongside
     it in the `max-width:1600px` block (`tokens.css:84-85`).

5. `#baseline-select` already has a second constraint.
   - It is `width:320px` **and** `max-width:40%` (`components.css:313`). The proposed
     `width: clamp(180px, 30cqi, 360px)` must drop/reconcile that `max-width:40%` or it caps the clamp.
   - `.race-progress` is `flex:0 0 42%` (not a raw width); `#race-banner` is `display:flex`, so the
     proposed `flex:1 1 320px` is a clean swap.

6. Second Plotly resize call-site.
   - Besides `app.js` (tab reveal), `boot.js:89` also resizes when the boot panel hides and the live
     dashboard appears (a hidden→visible transition). Keep **both** existing call-sites as fallbacks; the new
     ResizeObserver coexists with them (it may miss `display:none`→visible cases).

7. Height-constrained behavior: allow scroll.
   - Decision: when the viewport is too short, let the live dashboard **scroll** rather than clip/hide KPI
     content. The old single-screen 1080p no-shift layout becomes the preferred layout, not a hard contract.
   - Note: `scripts/ui_shot.py` defaults to a `1456x1086` viewport and `.claude/PODTERM_UI_TARGET.md`
     references the no-scroll baseline; re-run/re-baseline those after the change.

8. Path note: the files live under the nested package dir, e.g.
   `/home/sysreq/stellar/podterm/podterm/static/css/...`.


## Target Behavior

1. KPI cards scale from available container width.
   - Wide screens can keep a dense single row.
   - Medium screens should wrap to two rows instead of compressing every card.
   - Narrow screens should use fewer columns with readable values.

2. KPI typography scales by card/container size, not only viewport width.
   - Values should stay legible and avoid clipping.
   - Subtext should wrap only where useful; otherwise ellipsize with tooltips.

3. Dashboard vertical regions adapt to available viewport height.
   - Preserve no-layout-shift within cards.
   - Let chart/bottom regions flex when height is constrained.
   - Avoid forcing important KPI content offscreen.

4. Dialogs scale to the actual viewport.
   - Launch/config dialogs should use `min()`, `max()`, and viewport-relative widths.
   - Dialog content should remain scrollable without exceeding screen height.

5. Charts resize when their containers resize.
   - Plotly should respond to actual panel size changes, not only tab changes.


## CSS Plan

1. Replace fixed KPI grid columns with adaptive columns.
   - Current:
     - `grid-template-columns: repeat(9, minmax(0, 1fr))`
   - Proposed:
     - Use `repeat(auto-fit, minmax(var(--kpi-card-min), 1fr))`.
     - Define `--kpi-card-min` with responsive values:
       - large: `150px`
       - medium: `170px`
       - small: `210px`
   - Set `#kpi-row { height: auto; min-height: var(--h-kpi); }`.
   - Allow wrapped KPI rows to take more vertical space when needed.

2. Add container queries for KPI rows.
   - Add `container-type: inline-size` to `#kpi-row` and `#diag-row`.
   - Use `@container` rules to adjust:
     - `.kpi-card` padding
     - `.kpi-value` font size
     - `.kpi-sub` font size/line clamp
     - `.kpi-foot` height
   - This reacts to the dashboard presentation width, not the full browser width.

3. Make KPI typography clamp-based.
   - Replace static `--fs-kpi` usage with a bounded expression:
     - `font-size: clamp(18px, 1.8cqi, 28px)`
   - Keep minimums high enough for readability.
   - Use `font-variant-numeric: tabular-nums` as today.

4. Add card-level overflow handling.
   - Keep `.kpi-value` single-line for numeric values.
   - Add a utility for long values:
     - `max-width: 100%`
     - `overflow: hidden`
     - `text-overflow: ellipsis`
   - For note states, allow `.kpi-note-text` to wrap to two lines.

5. Adapt diagnostics row.
   - Replace `repeat(5, minmax(0, 1fr))` with `auto-fit`.
   - Use a separate `--diag-card-min`.
   - Let diagnostics row height become `auto` with a minimum.

6. Make the live dashboard vertical stack responsive.
   - Keep the fixed-height no-shift layout for tall 1080p+ viewports.
   - Add `@media (max-height: 900px)`:
     - reduce `--h-chart`
     - reduce `--h-bottom`
     - reduce vertical gaps
   - Add `@media (max-height: 760px)`:
     - allow page scroll over forcing hidden content
     - make bottom panels stack or shrink.

7. Improve race and baseline row scaling.
   - Let `#race-banner` wrap or switch to column layout on narrow containers.
   - Change `.race-progress` from fixed `42%` to `flex: 1 1 320px`.
   - Change `#baseline-select` from fixed `320px` to `width: clamp(180px, 30cqi, 360px)`.
   - Ensure target text can wrap or line-clamp on narrow widths.

8. Make dialogs responsive.
   - Replace `width: 520px` with:
     - `width: min(720px, calc(100vw - 32px))`
     - `max-height: min(80vh, calc(100dvh - 32px))`
   - Add narrower padding at small widths:
     - `padding: clamp(14px, 3vw, 24px)`
   - Give `#config-dialog` a wider cap:
     - `width: min(960px, calc(100vw - 32px))`
   - Keep internal `pre` scrollable.


## JavaScript Plan

1. Add a central chart resize observer.
   - Create a small helper such as `observePlotlyResize(root)`.
   - Attach `ResizeObserver` to `.chart-box`, `.chart-plot`, and compare chart containers.
   - Debounce `Plotly.Plots.resize(...)` with `requestAnimationFrame`.

2. Call the observer after charts are created.
   - In `charts.js` after `Plotly.newPlot`.
   - In `compare.js` after compare charts are rendered.

3. Avoid per-card JS sizing unless CSS cannot handle it.
   - Prefer container queries and `clamp()`.
   - Only add JS text fitting if a known KPI value still clips after CSS changes.

4. Keep card DOM stable.
   - Preserve the stable slots in `cards.js`.
   - Do not dynamically add/remove card sections based on size.
