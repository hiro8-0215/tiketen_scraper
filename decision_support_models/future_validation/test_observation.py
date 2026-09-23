import unittest
import pandas as pd
from observation import covered, demand_label
from strict_summary import compare


class ObservationTests(unittest.TestCase):
    def test_stale_listing_is_unknown(self):
        start=pd.Timestamp('2026-09-13')
        later=pd.Series({'status':'listing','last_observed_at':start})
        self.assertEqual(demand_label(None,later,start,1),-1)

    def test_confirmed_listing_is_active(self):
        start=pd.Timestamp('2026-09-13')
        later=pd.Series({'status':'listing','last_observed_at':start+pd.Timedelta(days=3)})
        self.assertEqual(demand_label(None,later,start,3),0)

    def test_outcome_after_deadline_does_not_prove_survival(self):
        start=pd.Timestamp('2026-09-13')
        later=pd.Series({'status':'deleted','last_observed_at':start+pd.Timedelta(days=5)})
        self.assertEqual(demand_label(None,later,start,1),-1)
        self.assertEqual(demand_label(None,later,start,7),2)

    def test_historical_sold_is_not_a_transition(self):
        start=pd.Timestamp('2026-09-13')
        later=pd.Series({'status':'sold','last_observed_at':start+pd.Timedelta(hours=2),
            'sold_at':start+pd.Timedelta(hours=2),'sold_at_source':'historical_unknown'})
        self.assertEqual(demand_label(None,later,start,1),-1)
        later['sold_at_source']='transition_observed'
        self.assertEqual(demand_label(None,later,start,1),1)

    def test_coverage_rejects_missing_failed_and_gapped_polls(self):
        start=pd.Timestamp('2026-09-13'); end=start+pd.Timedelta(days=1)
        records=[{'event_id':'e','complete':True,'observed_at':str(t)} for t in pd.date_range(start,end,freq='h')]
        self.assertTrue(covered(records,'e',start,end))
        self.assertFalse(covered(records,'other',start,end))
        self.assertFalse(covered([], 'e',start,end))
        self.assertFalse(covered(records[:5]+records[9:],'e',start,end))
        records[-1]['complete']=False
        self.assertFalse(covered(records,'e',start,end))

    def test_calibration_does_not_fit_single_date(self):
        f=pd.DataFrame({'landmark_at':[pd.Timestamp('2026-09-13')],'true_state':[0]})
        self.assertEqual(compare(f,'demand',1)['status'],'insufficient_evidence')

    def test_calibration_purges_overlapping_label_windows(self):
        f=pd.DataFrame({'landmark_at':pd.to_datetime(['2026-09-13','2026-09-14','2026-09-15']),
            'ticket_id':['a','b','c'],'true_state':[0,1,2]})
        result=compare(f,'demand',3)
        self.assertEqual(result['status'],'insufficient_evidence')
        self.assertEqual(result['counts'],{})



if __name__=='__main__':
    unittest.main()
