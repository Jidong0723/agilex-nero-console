import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from supervisor.pi05_adapter import Pi05InputAdapter
from test_pi05_adapter import _Broker,_InitiallyUnavailablePolicy
class LocalH10Tests(unittest.TestCase):
    def test_local_h10_profile_survives_cloud_and_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config/pi05.json';p.parent.mkdir();p.write_text((Path(__file__).parents[1]/'config/pi05.json').read_text())
            with patch('supervisor.pi05_adapter.OpenPIClient',_InitiallyUnavailablePolicy):a=Pi05InputAdapter(_Broker(),p)
            a._connection_stop.set();a._connection_wakeup.set();a._connection_thread.join(2)
            try:
                for profile in ('autodl','local5090'):
                    a.update_config({'inference_profile':profile})
                    self.assertEqual(a.config['execution']['expected_chunk_steps'],10)
                    self.assertEqual(a.config['execution']['replan_steps'],5)
                self.assertEqual(a.config['model']['checkpoint_label'],'NERO 11649 H10')
                self.assertEqual(a.config['model']['port'],8000)
            finally:a.close()
if __name__=='__main__':unittest.main()
