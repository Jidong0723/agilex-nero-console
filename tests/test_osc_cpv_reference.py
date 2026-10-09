"""CPV sender/reference contract, independent of CAN and numerical libraries."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from motion.osc_cpv_reference import CpvReference
from supervisor.authority import HardwareTxOwner


def command(reference_id='session-a', anchor=.1):
    return dict(reference_id=reference_id, anchor_position_rad=[anchor]*7,
                dt_s=.02, lower_rad=[-2.0]*7, upper_rad=[2.0]*7)


class CpvReferenceTests(unittest.TestCase):
    def prepare(self, reference, t, v=.5, epoch=1, meta=None):
        return reference.prepare(meta or command(), [v]*7, started_perf_ns=t,
                                 epoch=epoch, max_speed=1.0, max_acceleration=2.0)

    def test_consistent_position_velocity_and_transactional_commit(self):
        reference = CpvReference()
        candidate = self.prepare(reference, 1_000_000_000)
        self.assertIsNone(reference.snapshot())
        self.assertAlmostEqual(candidate['velocity_rad_s'][0], .04)
        self.assertAlmostEqual(candidate['position_rad'][0], .1008)
        reference.commit(candidate)
        for t in (1_015_000_000,1_040_000_000,1_071_000_000):
            previous = reference.snapshot()
            candidate = self.prepare(reference,t)
            dt=(t-previous['started_perf_ns'])/1e9
            for q,v,old,old_v in zip(candidate['position_rad'],candidate['velocity_rad_s'],
                                   previous['position_rad'],previous['velocity_rad_s']):
                self.assertAlmostEqual(q-old,v*dt)
                self.assertLessEqual(abs(v-old_v),2*dt+1e-10)
            reference.commit(candidate)

    def test_slow_feedback_does_not_reset_position_each_cycle(self):
        reference=CpvReference()
        for i in range(10):
            # The same anchor is deliberately attached to all proposals.
            # Only a new session/HOLD reference id may make it authoritative.
            candidate=self.prepare(reference,1_000_000_000+i*20_000_000)
            reference.commit(candidate)
        self.assertGreater(reference.snapshot()['position_rad'][0],.14)
        snapshot=reference.snapshot();snapshot['position_rad'][0]=999
        self.assertLess(reference.snapshot()['position_rad'][0],1)

    def test_delayed_send_does_not_extrapolate_a_single_position_step(self):
        reference=CpvReference()
        first=self.prepare(reference,1_000_000_000,v=.04)
        reference.commit(first)
        next_step=self.prepare(reference,1_040_000_000,v=.04)
        self.assertAlmostEqual(next_step['position_rad'][0]-first['position_rad'][0],.04*.02)
        self.assertAlmostEqual(next_step['velocity_rad_s'][0],.02)

    def test_epoch_and_hold_resume_reset_but_target_updates_do_not(self):
        reference=CpvReference()
        reference.commit(self.prepare(reference,1_000_000_000))
        next_step=self.prepare(reference,1_020_000_000,meta=command(anchor=.8))
        self.assertLess(next_step['position_rad'][0],.11)
        fresh=self.prepare(reference,1_020_000_000,meta=command('resumed',.8))
        self.assertAlmostEqual(fresh['position_rad'][0],.8008)
        fresh=self.prepare(reference,1_020_000_000,epoch=2,meta=command(anchor=.7))
        self.assertAlmostEqual(fresh['position_rad'][0],.7008)

    def test_safety_gate_is_not_undone_and_position_limit_is_checked(self):
        reference=CpvReference()
        reference.commit(self.prepare(reference,1_000_000_000))
        gated=command();gated['gate_mask']=[True]*7
        stopped=self.prepare(reference,1_001_000_000,v=0,meta=gated)
        self.assertEqual(stopped['velocity_rad_s'],[0.0]*7)
        self.assertEqual(stopped['position_rad'],reference.snapshot()['position_rad'])
        bounded=command(anchor=1.9999)
        with self.assertRaisesRegex(RuntimeError,'outside joint limits'):
            self.prepare(CpvReference(),1_000_000_000,meta=bounded)

    def test_bad_interval_and_nonfinite_velocity_do_not_commit(self):
        reference=CpvReference()
        reference.commit(self.prepare(reference,1_000_000_000))
        old=reference.snapshot()
        for t in (1_000_000_000,1_201_000_000):
            with self.assertRaises(RuntimeError):self.prepare(reference,t)
        with self.assertRaises(ValueError):self.prepare(reference,1_020_000_000,v=float('nan'))
        self.assertEqual(reference.snapshot(),old)


class CpvReferenceSenderTests(unittest.TestCase):
    def setUp(self):
        self.backend=SimpleNamespace(send_cpv_position=Mock(return_value={'ok':True}))
        self.owner=HardwareTxOwner(self.backend)
        self.owner.close()  # deterministic manual scheduling of the real dispatch
        self.clock=1_000_000_000
        self.revision=0

    def entry(self,v=.5,reference_id='session-a'):
        self.revision+=1
        return dict(mailbox_revision=self.revision,epoch=0,target_generation=self.revision,
                    control_sample_id=self.revision,joint_target_rad=[.1]*7,
                    joint_velocity_rad_s=[v]*7,cpv_reference=command(reference_id),
                    max_joint_speed_rad_s=1.0,max_joint_acceleration_rad_s2=2.0,
                    published_monotonic_ns=1,gate_ok=True)

    def dispatch(self,entry,dt_ns=20_000_000):
        self.clock+=dt_ns
        with patch('supervisor.authority.time.perf_counter_ns',return_value=self.clock):
            self.owner._dispatch_cpv(entry)
        return self.owner.cpv_diagnostics()['last_result']

    def test_send_jitter_and_stale_position_proposal_never_rewind_reference(self):
        previous=None
        for dt in (20_000_000,31_000_000,17_000_000,24_000_000):
            entry=self.entry()
            entry['joint_target_rad']=[-.8]*7  # obsolete proposal, not history
            sent=self.dispatch(entry,dt)
            self.assertEqual(sent['status'],'sent')
            if previous:
                elapsed=(sent['dispatch_started_perf_ns']-previous['dispatch_started_perf_ns'])/1e9
                self.assertAlmostEqual(sent['joint_target_rad'][0]-previous['joint_target_rad'][0],
                                       sent['joint_velocity_rad_s'][0]*elapsed)
            previous=sent
        self.assertGreater(sent['joint_target_rad'][0],.1)

    def test_failed_or_revoked_send_never_advances_reference(self):
        self.dispatch(self.entry())
        old=copy.deepcopy(self.owner.cpv_diagnostics()['cpv_reference'])
        self.backend.send_cpv_position.return_value={'ok':False}
        self.assertEqual(self.dispatch(self.entry())['status'],'failed')
        self.assertEqual(self.owner.cpv_diagnostics()['cpv_reference'],old)
        entry=self.entry();entry['execute_guard']=lambda:False
        self.assertEqual(self.dispatch(entry)['status'],'revoked')
        self.assertEqual(self.owner.cpv_diagnostics()['cpv_reference'],old)

    def test_rejected_safety_gate_cannot_send_or_commit(self):
        entry=self.entry();entry['gate_ok']=False
        self.assertEqual(self.dispatch(entry)['status'],'failed')
        self.backend.send_cpv_position.assert_not_called()
        self.assertIsNone(self.owner.cpv_diagnostics()['cpv_reference'])

    def test_50hz_producer_and_delayed_sender_only_integrate_successful_batches(self):
        # Targets update at 50 Hz; solves finish off-phase and sends have
        # alternating intervals. Mailbox replacement must not integrate work
        # which was never sent or re-anchor at each target generation.
        ready=[];sent_times=[];prior=None
        for ms in range(1001):
            if ms%20==0:
                ready.append((ms+(7 if ms%40 else 29),self.entry()))
            for due,entry in list(ready):
                if due<=ms:
                    self.owner.publish_cpv(entry)
                    ready.remove((due,entry))
            if ms%37==11:
                entry=self.owner._take_cpv()
                if entry:
                    self.clock=1_000_000_000+ms*1_000_000
                    with patch('supervisor.authority.time.perf_counter_ns',return_value=self.clock):
                        self.owner._dispatch_cpv(entry)
                    sent=self.owner.cpv_diagnostics()['last_result']
                    self.assertEqual(sent['status'],'sent')
                    if prior:
                        dt=(sent['dispatch_started_perf_ns']-prior['dispatch_started_perf_ns'])/1e9
                        self.assertAlmostEqual(sent['joint_target_rad'][0]-prior['joint_target_rad'][0],
                                               sent['joint_velocity_rad_s'][0]*dt)
                    prior=sent;sent_times.append(ms)
        diag=self.owner.cpv_diagnostics()
        self.assertGreater(len(sent_times),20)
        self.assertLessEqual(max(b-a for a,b in zip(sent_times,sent_times[1:])),74)
        self.assertEqual(diag['failed_count'],0)
        self.assertEqual(diag['revoked_count'],0)
        self.assertGreater(diag['superseded_count'],0)
        self.assertGreater(diag['cpv_reference']['position_rad'][0],.3)

    def test_stop_barrier_and_new_epoch_prevent_old_reference_restoration(self):
        self.dispatch(self.entry())
        stale=self.entry()
        self.owner.revoke_cpv_before_generation(0,stale['target_generation']+1,'HOLD')
        self.assertEqual(self.dispatch(stale)['status'],'revoked')
        self.owner.advance_epoch(1)
        stale=self.entry()
        self.assertEqual(self.dispatch(stale)['status'],'revoked')
        new=self.entry(reference_id='session-b');new['epoch']=1
        new['cpv_reference']['anchor_position_rad']=[.6]*7
        sent=self.dispatch(new)
        self.assertEqual(sent['status'],'sent')
        self.assertAlmostEqual(sent['joint_target_rad'][0],.6008)


if __name__=='__main__':
    unittest.main()
