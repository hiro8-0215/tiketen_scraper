# Listing evidence and retained history

## Full public content

The event list refreshes active inventory, price, quantity, date, venue and
available seat and tag fields. Listing descriptions are previews, saved in
`api_description`. Only an identity-verified public detail response sets
`description_source=public_detail`, `description_is_full=True` and
`description_checked_at`. Full public text can legitimately be empty.

The details phase fetches missing full text first, refreshes a changed preview
when it no longer matches the saved full text, and rechecks saved full text
after 24 hours. An edit beyond an unchanged preview is picked up by the
24-hour recheck. A failed request leaves the last verified full text and its
original verification timestamp intact. Prices and other fields available in
the list still follow the latest list observation. The CSV columns and history
formats are unchanged.

The public API requests are spaced across both workers, and HTTP 429 waits are
shared and carried to the next phase within the same run. Checkpoints allow
later runs to continue with unverified rows. Regular Actions polls use a
5-minute shared deadline for missing-ID lookups and 8 minutes for descriptions.
Inventory collection continues for all configured events after the individual
lookup deadline. Per-group and per-event priority favors the least recently
attempted status work, so a large event cannot always take the whole budget.
For initial collection or a large backlog, manually select
`recover_details` in Ticket Scraper for a longer detail pass, or use
`SCRAPE_MODE=details` with `PUBLIC_DETAIL_RUNTIME_MINUTES` locally. The recovery
run uses the same request pacing and honors the site's retry instructions.

`content_observation_*.jsonl` records each batch's full/failed/null/pending
counts. The regular workflow validates the current inventory and publishes the
pending individual lookups separately in the Actions summary
(`public_coverage_complete=false` while they remain). Backup copies all
currently saved masters, including last verified text and explicit provenance.
Listing coverage, structural errors and failed requests still fail the run.
The longer recovery mode uses `tools/audit_public_content.py --data-dir data
--rolling` to require verified active descriptions and completed recent state
checks. GitHub persists the collected data before surfacing audit errors.
Only a successful scrape starts the automatic Drive archive.

The collector never resets masters. Daily public share codes can resolve to a
different canonical share code through the same public lookup as the ticket
page. Only this explicit response (or a previously verified creation identity)
links IDs; matching descriptions/prices never does. Original rows are retained
as `observation_state=alias` with `canonical_ticket_id`. The linked lifecycle's
earliest observed timestamp is `identity_first_observed_at`; original ID
timestamps remain unchanged. Linked current codes use
`first_observed_source=public_alias_observed`, not a new-listing label.

`status` on old rows is the last confirmed historical state. Unmatched rows are
`absent_unverified`; a successful public null response is `absent_unknown`, not
proof of sale/withdrawal. Transport failures remain retryable.
`inactive` is a confirmed unavailable state, not a sale/withdrawal, and remains
retryable; it is excluded from current active supply. When an individual public
response includes a numeric price, preserve that explicitly observed price
rather than an older listing price (including for confirmed sales).
Sold/cancelled individual public responses can confirm a transition; anonymous API sold rows
remain separate. Sold timestamps are first confirmation times, not exact
transaction times. A public null or expired code cannot reconstruct lost history.

Current market counts exclude aliases and unverified/unknown absences. Price on
request is recorded as `is_price_on_request=True` / `price_source=on_request`;
price zero is not a zero-yen transaction. `ticket_changes_YYYYMMDD.jsonl` retains
changed price, quantity, state and ID linkage facts even when masters are
overwritten on the same day. Unchanged polls do not duplicate change records.

Public checks prioritize missing IDs seen within 36 hours. The default limit
is 1500 checks per run, at most two concurrent requests with delay and a 15-minute
budget. `PUBLIC_STATUS_CHECK_LIMIT` controls the count. Pending, failed and
historically unresolved counts appear in `observation_*.jsonl`. Keep hourly
collection so valid old codes can be reconciled before they expire.

Downstream models must respect aliases/canonical IDs and lifecycle first-seen
timestamps, rather than count every new public code as a new listing. Existing
trained models are not automatically updated by this collection change.

Google Drive CSV backups include the added columns. JSONL upload remains
opt-in: redeploy `google_drive_apps_script/Code.gs`, then enable
`GDRIVE_INCLUDE_JSONL=true`. Without this, logs remain on GitHub, not Drive.

Verified 2026-09-27: offline tests and isolated public API probes retained all
original master IDs. Some old codes resolve to today's canonical code; others
have no public evidence. The latter are not merged or given invented outcomes.
