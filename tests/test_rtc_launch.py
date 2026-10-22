import errno
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock


SCRIPT=Path(__file__).parents[1]/'inference/rtc/start_production.sh'


class ProductionLaunchTests(unittest.TestCase):
    def test_port_check_never_connects_and_fails_closed(self):
        source=SCRIPT.read_text(encoding='utf-8').split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
        for error in (None,OSError(errno.EADDRINUSE,'used'),OSError(errno.EACCES,'denied')):
            probe=MagicMock();probe.__enter__.return_value=probe;probe.bind.side_effect=error
            with patch('socket.socket',return_value=probe):
                if error is None:exec(compile(source,'production_port_check','exec'),{})
                elif error.errno==errno.EADDRINUSE:
                    with self.assertRaises(SystemExit) as caught:exec(compile(source,'production_port_check','exec'),{})
                    self.assertEqual(caught.exception.code,10)
                else:
                    with self.assertRaises(OSError):exec(compile(source,'production_port_check','exec'),{})
                probe.connect.assert_not_called()

    def test_one_pinned_entry_not_old_environment_profile_chain(self):
        source=SCRIPT.read_text(encoding='utf-8')
        for expected in ('--steps 20 --normal-steps 10 --guidance-cap 1000',
                         '--prefix-attention-schedule exp --prefix-mask-mode timestamp_frozen',
                         '--warmup-inference-delay 6 --warmup-execute-horizon 6',
                         '--terminal-noise-time 0.001 --terminal-refinement-steps 10'):
            self.assertIn(expected,source)
        self.assertNotIn('start_terminal_refined.sh',source)
        self.assertNotIn('${RTC_',source)


if __name__=='__main__':unittest.main()
