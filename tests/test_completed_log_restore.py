import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import io

spec = importlib.util.spec_from_file_location('recording', Path(__file__).parents[1]/'supervisor/inference_recording.py')
recording = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recording)


class CompletedLogRestoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_files(self, name, **override):
        path = self.root/name
        path.mkdir()
        summary = dict(run_id=name, finished=True, export_ready=True, write_error=None, lost_records=0)
        summary.update(override)
        (path/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        (path/'events.jsonl').write_text('{"event":"run_start"}\n{"event":"run_end"}\n', encoding='utf-8')
        return path

    def test_no_history(self):
        self.assertIsNone(recording.latest_completed_run(self.root))
        self.assertIsNone(recording.latest_completed_run(self.root/'absent'))

    def test_restore_latest_after_restart_and_export_exact_content(self):
        self.run_files('20261027T100000-aaaaaaaa')
        path = self.run_files('20261027T110000-bbbbbbbb')
        for _ in range(2):
            restored = recording.latest_completed_run(self.root)
            self.assertEqual(restored.id, path.name)
            name, payload = restored.zip_bytes()
            self.assertEqual(name, path.name+'.zip')
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                self.assertEqual(set(archive.namelist()), {'events.jsonl','summary.json'})
                for file in archive.namelist():
                    self.assertEqual(archive.read(file), (path/file).read_bytes())

    def test_incomplete_and_malformed_not_exported(self):
        valid = self.run_files('20261027T100000-aaaaaaaa')
        self.run_files('20261027T110000-bbbbbbbb', finished=False)
        bad = self.run_files('20261027T120000-cccccccc')
        (bad/'summary.json').write_text('{', encoding='utf-8')
        self.assertEqual(recording.latest_completed_run(self.root).id, valid.name)

    def test_recorded_loss_does_not_fall_back_or_export(self):
        self.run_files('20261027T100000-aaaaaaaa')
        bad = self.run_files('20261027T110000-bbbbbbbb', lost_records=1)
        restored = recording.latest_completed_run(self.root)
        self.assertEqual(restored.id, bad.name)
        self.assertFalse(restored.status()['export_ready'])
        with self.assertRaises(RuntimeError):
            restored.zip_bytes()

    def test_missing_events_disabled_and_summary_immutable(self):
        path = self.run_files('20261027T100000-aaaaaaaa')
        (path/'events.jsonl').unlink()
        restored = recording.latest_completed_run(self.root)
        self.assertFalse(restored.status()['export_ready'])
        snapshot = restored.status()
        snapshot['export_ready'] = True
        self.assertFalse(restored.status()['export_ready'])

    def test_finished_write_error_disabled(self):
        self.run_files('20261027T100000-aaaaaaaa', write_error='disk full')
        self.assertFalse(recording.latest_completed_run(self.root).status()['export_ready'])


if __name__ == '__main__':
    unittest.main()
