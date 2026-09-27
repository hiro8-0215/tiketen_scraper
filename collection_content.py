"""Separate event previews from timestamped, identity-verified public content."""
import json
import math

CONTENT_COLUMNS = ['api_description', 'description_source', 'description_checked_at',
                   'description_is_full', 'content_checked_at', 'public_event_id', 'seat_type']


def apply_api_preview(row, ticket):
    if 'description' not in ticket or ticket['description'] is None:
        return
    preview = ticket['description']
    if not isinstance(preview, str):
        raise ValueError('Invalid API description')
    row['api_description'] = preview
    # Neither a legacy browser full text nor a verified public text is replaced.
    verified_full = (row.get('description_source') == 'public_detail'
                     and row.get('description_is_full') == 'True')
    if not verified_full and (not row.get('raw_description')
                              or row.get('description_source') == 'event_api_preview'):
        row['raw_description'] = preview
        row['description_source'] = 'event_api_preview'
        row['description_is_full'] = 'False'


def public_updates(ticket, now):
    """Validate before mutation; missing optional fields retain earlier facts."""
    updates = {}
    if 'quantity' in ticket and ticket['quantity'] is not None:
        value = ticket['quantity']
        if isinstance(value, bool):
            raise ValueError('Invalid public quantity')
        number = float(value)
        if not math.isfinite(number) or number < 1 or not number.is_integer():
            raise ValueError('Invalid public quantity')
        updates['quantity'] = int(number)
    for field, key in [('perf_date','eventDate'), ('perf_time','eventStartTime'),
                       ('venue','venue'), ('ticket_type','ticketType'),
                       ('name_type','nameGender'), ('delivery_method','deliveryMethod')]:
        if key in ticket and ticket[key] is not None:
            if not isinstance(ticket[key], str):
                raise ValueError('Invalid public field: ' + key)
            updates[field] = ticket[key]
    if 'description' in ticket and ticket['description'] is not None:
        if not isinstance(ticket['description'], str):
            raise ValueError('Invalid public description')
        updates.update(raw_description=ticket['description'], description_source='public_detail',
                       description_checked_at=now, description_is_full='True', details_fetched='True')
    if 'seatType' in ticket:
        updates['seat_type'] = json.dumps(ticket['seatType'], ensure_ascii=True, sort_keys=True)
    seller = ticket.get('sellerInfo')
    if isinstance(seller, dict):
        if isinstance(seller.get('displayName'), str):
            updates['seller_name'] = seller['displayName']
        if seller.get('rating') is not None:
            updates['seller_rating'] = seller['rating']
    elif isinstance(ticket.get('sellerName'), str):
        updates['seller_name'] = ticket['sellerName']
    tags = ticket.get('tags')
    if isinstance(tags, list):
        labels = [tag.get('label', '') if isinstance(tag, dict) else tag
                  for tag in tags]
        if all(isinstance(label, str) for label in labels):
            updates['ticket_tags'] = ' / '.join(label for label in labels if label)
    if updates:
        updates['content_checked_at'] = now
    if ticket.get('eventId'):
        updates['public_event_id'] = ticket['eventId']
    return updates
