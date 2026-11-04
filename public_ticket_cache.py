"""Reuse verified public replies between phases of one workflow execution."""
import json
import os
from pathlib import Path


def remember_verified(cache, ticket, error, checked_at):
    """Call only after identity/event validation by apply_public_evidence."""
    if ticket is None or error or not checked_at:
        return
    code = ticket['shareCode']
    previous = cache.get(code)
    if previous is None or previous[1] or previous[2] <= checked_at:
        cache[code] = (ticket, None, checked_at)


def load_run_cache():
    filename = os.environ.get('PUBLIC_TICKET_CACHE_PATH')
    if not filename or not Path(filename).exists():
        return {}
    with Path(filename).open(encoding='utf-8') as stream:
        payload = json.load(stream)
    return {code: tuple(value) for code, value in payload.items()
            if isinstance(code, str) and isinstance(value, list) and len(value) == 3
            and isinstance(value[0], dict) and value[1] is None and isinstance(value[2], str)}


def save_run_cache(cache):
    filename = os.environ.get('PUBLIC_TICKET_CACHE_PATH')
    if not filename:
        return
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump({code: value for code, value in cache.items()
                       if value[0] is not None and value[1] is None}, stream, ensure_ascii=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
