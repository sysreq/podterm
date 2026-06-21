# Screen Scaling Implementation Plan

## Implementation Steps

1. Add responsive sizing tokens.
   - Add `--kpi-card-min`, `--diag-card-min`, and responsive height variants in `tokens.css`.
   - Replace the single `max-width: 1600px` breakpoint with width and height ranges.

2. Update layout grids.
   - Change `#kpi-row` and `#diag-row` to adaptive grids.
   - Let wrapped rows expand naturally.
   - Verify the live stack still scrolls when rows wrap.

3. Update KPI card CSS.
   - Add container query behavior.
   - Use clamp-based value/subtext sizing.
   - Improve note wrapping and long-value handling.

4. Update race/baseline scaling.
   - Make race progress and baseline select flexible.
   - Add narrow-container layout rules.

5. Update dialog CSS.
   - Replace fixed dialog width.
   - Add per-dialog caps for launch versus full-config JSON.

6. Add Plotly resize observation.
   - Implement a reusable observer.
   - Wire live and compare charts.
   - Keep existing tab-reveal resize as a fallback.

7. Visual verification.
   - Capture/check at these viewport sizes:
     - `1366x768`
     - `1440x900`
     - `1920x1080`
     - `2560x1440`
     - narrow presentation width around `1100x900`
   - Check browser zoom at `125%` and `150%`.
   - Confirm no KPI value overlaps, no important text is clipped without tooltip, and charts resize cleanly.


## Risks And Constraints

1. Wrapping KPI rows changes vertical space.
   - This is intentional for readability, but it means the old exact 1080p height equation should become a preferred layout, not a hard contract.

2. Container queries need modern browser support.
   - This is acceptable for current Chromium/Safari/Firefox.
   - If support is a concern, keep viewport media queries as a fallback.

3. Plotly can render incorrectly when initialized in hidden or zero-width containers.
   - Keep existing tab reveal resizing.
   - Add resize observation after chart creation.

4. Too much automatic font scaling can make the dashboard feel inconsistent.
   - Use bounded `clamp()` values and keep fixed minimums.
