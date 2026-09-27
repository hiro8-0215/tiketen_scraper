# Listing evidence and retained history

## Full public content (2026-09-27 correction)

Event API descriptions are previews, stored as `api_description`. They never
overwrite a saved full `raw_description`. The details stage now refreshes all
confirmed active tickets through the same public lookup used by ticket pages,
with two workers, 0.5s per-request delay and the existing 25-minute checkpoint
budget. Repeated performer copies share the same response cache. Each full
description has `description_source=public_detail`, `description_is_full=True`
and `description_checked_at`; `content_checked_at` timestamps field refreshes.
Public responses also refresh quantity, date/time, venue, ticket/name/delivery
type, seat metadata, seller rating/name and tags, including inactive/paused
responses checked by the API stage. Old ID observation timestamps do not change.
Null, failed or stale replies do not erase earlier text. Never overwrite an
earlier historical snapshot with today's description.

`content_observation_*.jsonl` reports full/failed/null/pending counts. The workflow
saves resumable data before running `tools/audit_public_content.py --data-dir data`;
incomplete current full content fails the quality step instead of silently
claiming success. Retained old unknown rows are not fabricated or counted as
currently verified full content. Full public text may legitimately be empty or
contain ellipses typed by the seller; provenance, not text length, proves coverage.

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
