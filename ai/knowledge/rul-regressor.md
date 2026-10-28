# Remaining useful life (RUL) regressor

The RUL model is experimental. It estimates hours remaining on failed IMS trajectories that were scored.

## What exists
- Stored field: `predicted_rul_hours` on windows where the regressor produced a value.
- Test last-24-hour MAE: 2 hours (the number MaintIQ quotes).
- Last-72-hour MAE is the other operational figure to quote.

## What not to quote
- Full-run MAE is dominated by early-life hundreds of hours (validation 208 h, test 403 h) and is the wrong headline.
- RUL is missing on bearings or windows that were not scored. Leave those as "not stored".

## Limitations
- Only failed, scored trajectories have RUL. Do not invent hours for uploads or for windows without a stored value.
- The model does not certify remaining service life for a production asset.
- Use it as a historical comparison on IMS runs, not as a go / no-go operating decision.
