# Listing evidence and retained history

## Full public content

The event list refreshes active inventory, price, quantity, date, venue and
available seat and tag fields. Listing descriptions are previews, saved in
`api_description`. Only an identity-verified public detail response sets
`description_source=public_detail`, `description_is_full=True` and
`description_checked_at`. Full public text can legitimately be empty.

The ordinary details phase checks every group for visible preview changes
first, then missing full text, then descriptions older than the latest
inventory observation. After a bulk code change, the state phase can confirm
the link and full text from one reply. The details phase then checks active
rows that still need full text in the same run. It shares priority
slots across groups within each band and saves replies already received
even if the request window closes. Full text becomes due for a periodic
recheck at 24 hours. An edit beyond an unchanged preview can only be seen
through a subsequent detail lookup; higher-priority work or rate limits can
delay that lookup beyond the 24-hour threshold. A failed request leaves the last verified full text and its
original verification timestamp intact. Prices and other fields available in
the list still follow the latest list observation. The CSV columns and history
formats are unchanged.

The ordinary Actions poll starts at a 4.5-second global interval after the
two-worker burst triggered HTTP 429 in production. It allows up to 30 minutes
for state and 40 minutes for details,
with a shared 53-minute deadline measured from job setup. The last seven
minutes of the hourly job are reserved for audit, packing and push. The optional
repository variable `PUBLIC_API_MIN_INTERVAL` adds spacing across workers.
On HTTP 429 or an HTTP 503 with a retry instruction, the collector waits and
sets global spacing to at least 4.5 seconds for the rest of the run.
Server cooldowns carry over to queued runs, and access denial stops further
individual requests in the same run. Actions records each phase's HTTP counts.
The collector saves inventory for every event first. It then checks newly
absent IDs across events before older unresolved IDs, sharing the available
requests among events in each priority band. Observation logs include `newly_missing`,
`newly_missing_pending`, and `full_from_alias`.

For a single-run attempt, manually select `recover_details` in Ticket Scraper.
That mode gives recent missing IDs up to 265 minutes, then uses the remaining
shared time for descriptions. After a mid-run HTTP 403 following valid results,
it allows one retry after a 15-minute cooldown. HTTP 401 and failed validation
surface as errors. Recovery shares the concurrency group with hourly
collection, so a long recovery can delay scheduled runs. The runner can finish
only the work that the site and the shared time budget allow.

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
The GitHub schedule fallback checks at minute 12 whether this hour already
has a Ticket Scraper run, including queued runs. It dispatches an ordinary
run for a missing hour and leaves the external hourly trigger in place.
The scraper concurrency group serializes a fallback run behind any previous run.
An anonymous sold row remains aggregate evidence even when the run verifies
all known recent listing outcomes. The summary reports those rows separately.
CSV evidence timestamps and log dates use UTC, including on a local PC.
Performance dates and the Drive folder calendar use Japan time.

If an event gains a new URL, matching public event IDs preserve the stored
event key and let the collector fetch its inventory once. A share code from a
different public event still fails validation.

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

Current market counts also exclude listings absent from the public inventory
even if a direct lookup still reports them as active. Price on
request is recorded as `is_price_on_request=True` / `price_source=on_request`;
price zero is not a zero-yen transaction. `ticket_changes_YYYYMMDD.jsonl` retains
changed price, quantity, state and ID linkage facts even when masters are
overwritten on the same day. Unchanged polls do not duplicate change records.

Public checks prioritize missing IDs seen within 36 hours. An eligible
`absent_unknown` code can be tried again after six hours while the absence
is recent. Pending, failed, and historically unresolved counts appear in
`observation_*.jsonl`.

Downstream models must respect aliases/canonical IDs and lifecycle first-seen
timestamps, rather than count every new public code as a new listing. Existing
trained models are not automatically updated by this collection change.

Google Drive CSV backups include the added columns. JSONL upload remains
opt-in: redeploy `google_drive_apps_script/Code.gs`, then enable
`GDRIVE_INCLUDE_JSONL=true`. Without this, logs remain on GitHub, not Drive.

Verified 2026-09-27: offline tests and isolated public API probes retained all
original master IDs. Some old codes resolve to today's canonical code; others
have no public evidence. The latter are not merged or given invented outcomes.
