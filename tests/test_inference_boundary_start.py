import json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from supervisor.pi05_adapter import Pi05InputAdapter
from test_pi05_adapter import _Broker,_Camera,_InitiallyUnavailablePolicy

class BoundaryStartTests(unittest.TestCase):
    def make(self,tmp):
        p=Path(tmp)/'config/pi05.json';p.parent.mkdir()
        p.write_text((Path(__file__).parents[1]/'config/pi05.json').read_text())
        with patch('supervisor.pi05_adapter.OpenPIClient',_InitiallyUnavailablePolicy):a=Pi05InputAdapter(_Broker(),p)
        # Keep startup-race coverage independent of the operator's live RTC selection.
        a.config['execution']['mode']='normal'
        a._connection_stop.set();a._connection_wakeup.set();a._connection_thread.join(2)
        a._osc_state_stop.set();a._osc_state_thread.join(2)
        return a
    def test_small_overshoot_clips_command_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.make(tmp)
            try:
                for value,expected in [(1.0025667560243994,1.),(-.0025,0.),(.5,.5)]:
                    original=[0,0,0,0,0,0,value]
                    _,actual=a._target_from_action(original,{'position_m':[0,0,.3],'orientation_xyzw':[0,0,0,1]},action_index=8)
                    self.assertEqual(actual,expected);self.assertEqual(original[-1],value)
                self.assertEqual(a.state['gripper_ratio_diagnostic']['raw_ratio'],-.0025)
            finally:a.close()
    def test_large_or_nonfinite_output_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.make(tmp)
            try:
                for value in (1.051,-.01,float('nan'),float('inf')):
                    with self.assertRaises((RuntimeError,ValueError)):
                        a._target_from_action([0,0,0,0,0,0,value],{'position_m':[0,0,.3],'orientation_xyzw':[0,0,0,1]})
            finally:a.close()
    def test_cancel_during_start_state_read_prevents_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.make(tmp);a.cameras=_Camera()
            entered=threading.Event();release=threading.Event();errors=[]
            original=a.osc.state
            def slow():entered.set();release.wait(2);return original()
            a.osc.state=slow
            def start():
                try:a.start('session-1','client-1')
                except Exception as e:errors.append(str(e))
            thread=threading.Thread(target=start);thread.start()
            try:
                self.assertTrue(entered.wait(1))
                a.stop('cancel startup');release.set();thread.join(2)
                self.assertTrue(errors);self.assertIn('cancelled',errors[0])
                self.assertFalse(a.state['execution_enabled']);self.assertIsNone(a._capture)
                self.assertFalse(a.osc.commands)
            finally:release.set();thread.join(2);a.close()

if __name__=='__main__':unittest.main()
