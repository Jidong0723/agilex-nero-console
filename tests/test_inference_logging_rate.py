import json,tempfile,threading,time,unittest
from supervisor.pi05_adapter import Pi05InputAdapter
from supervisor.inference_recording import RunCapture

class LoggingRateTests(unittest.TestCase):
    def adapter(self):
        a=Pi05InputAdapter.__new__(Pi05InputAdapter)
        a.lock=threading.RLock();a._log_feedback_snapshot={}
        return a

    def frame(self,n):
        return {'session':{'state':'ACTIVE','id':'s'},'execution':{
            'feedback_revision':n,'control_sample_id':n,
            'measured_tcp_pose':{'position_m':[n*.001,0,0],'orientation_xyzw':[0,0,0,1]}},
            'transport':{'last_cpv_dispatch':{'control_sample_id':n,'status':'sent','ok':True}}}

    def test_sampling_continues_while_inference_lock_held(self):
        a=self.adapter();a._publish_logging_feedback(self.frame(1))
        with tempfile.TemporaryDirectory() as tmp:
            with a.lock:
                log=RunCapture(tmp,{},a._logging_feedback)
                time.sleep(.6)
                self.assertGreaterEqual(log.sample_count,25)
            log.finish('test')
            status=log.status()
            self.assertGreater(status['actual_record_hz'],40)
            self.assertEqual(status['fresh_feedback_count'],1)
            self.assertGreaterEqual(status['reused_feedback_count'],24)
            self.assertLess(status['max_cache_read_ms'],20)

    def test_new_publications_are_consistent_and_not_mutable(self):
        a=self.adapter();source=self.frame(4);a._publish_logging_feedback(source)
        source['execution']['feedback_revision']=99
        frame=a._logging_feedback();self.assertEqual(frame['execution']['feedback_revision'],4)
        frame['execution']['feedback_revision']=88
        self.assertEqual(a._logging_feedback()['execution']['feedback_revision'],4)
        self.assertIsNotNone(frame['cache_published_perf_ns'])

    def test_freshness_and_monotonic_event_order(self):
        a=self.adapter();a._publish_logging_feedback(self.frame(1))
        with tempfile.TemporaryDirectory() as tmp:
            log=RunCapture(tmp,{},a._logging_feedback)
            for n in range(2,12):
                a._publish_logging_feedback(self.frame(n));time.sleep(.025)
                log.emit('model_chunk',actions=[[.1]*7]*20)
            log.finish('test')
            rows=[json.loads(s) for s in (log.path/'events.jsonl').read_text().splitlines()]
            self.assertEqual([r['perf_ns'] for r in rows],sorted(r['perf_ns'] for r in rows))
            self.assertGreaterEqual(log.fresh_sample_count,8)
            self.assertTrue(all('sample_started_perf_ns' in r for r in rows if r['event']=='feedback_sample'))
            self.assertEqual(log.lost,0);self.assertIsNone(log.error)

    def test_slow_cache_reports_missed_periods_without_catchup_burst(self):
        def slow():time.sleep(.06);return self.frame(1)
        with tempfile.TemporaryDirectory() as tmp:
            log=RunCapture(tmp,{},slow);time.sleep(.27);log.finish('test')
            self.assertGreater(log.status()['missed_sampling_periods'],0)
            self.assertGreater(log.status()['max_cache_read_ms'],50)
            self.assertLess(log.sample_count,8)

if __name__=='__main__':unittest.main()
