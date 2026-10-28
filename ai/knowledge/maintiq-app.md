# MaintIQ application map

## Pages a user can open
- Dashboard: fleet cards from stored machines and alerts. Health-trend chart has a machine dropdown.
- Machines: inventory, search, filters, edit user fields, notes, add machine with CSV/ZIP/TXT upload.
- Machine details: one asset, scores if present, notes, upload metadata.
- Monitoring: scoring runs (Queued, Running, Paused, Completed, Failed). IMS seeds are Completed.
- Alerts: Isolation Forest and classifier flags. Statuses: New, Acknowledged, Investigating, Resolved. Severities: Critical, High, Medium, Low.
- Analytics: vibration, anomalies, failure probability, frequency, model performance. Filters: machine, from, to, processed features.
- Reports: stored condition assessments for IMS Bearing 3 (miq-rpt-001) and IMS Bearing 4 (miq-rpt-002).
- AI Assistant: this chat. Settings: profile and password.

## Navigation destinations the assistant may request
- dashboard, machines, machine-details, analytics, alerts, monitoring, reports
- machine-details and analytics need a real public machine id (for example brg-003).
- reports may include report id miq-rpt-001 or miq-rpt-002.

## Uploads
Accepted files: csv, zip, txt. Validation records row/column counts. No model scores are invented for a new upload.
