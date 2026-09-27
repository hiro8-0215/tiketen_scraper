import csv
import tempfile
import unittest
from pathlib import Path
from tools.audit_collection_run import audit


class CollectionRunAuditTest(unittest.TestCase):
    def write(self, folder, rows):
        path = folder / 'artist_master.csv'
        fields = sorted({key for row in rows for key in row})
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_retained_history_and_proved_alias_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            before, after = Path(temp)/'before', Path(temp)/'after'
            before.mkdir(); after.mkdir()
            old = {'ticket_id':'old','event_id':'event','status':'listing',
                   'first_observed_at':'2026-09-25 01:00:00'}
            self.write(before, [old])
            self.write(after, [{**old,'canonical_ticket_id':'new','observation_state':'alias'},
                              {'ticket_id':'new','event_id':'event','status':'listing',
                               'canonical_ticket_id':'new','observation_state':'active'}])
            report = audit(before, after)
            self.assertTrue(report['ok'])
            self.assertEqual(report['totals']['lost_original_ids'], 0)
            self.assertEqual(report['totals']['alias_rows'], 1)

    def test_lost_id_and_unproven_transition_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            before, after = Path(temp)/'before', Path(temp)/'after'
            before.mkdir(); after.mkdir()
            self.write(before, [{'ticket_id':'lost','status':'listing'},
                                {'ticket_id':'changed','status':'listing'}])
            self.write(after, [{'ticket_id':'changed','status':'sold'}])
            report = audit(before, after)
            self.assertFalse(report['ok'])
            self.assertEqual(len(report['errors']), 2)


if __name__ == '__main__':
    unittest.main()
