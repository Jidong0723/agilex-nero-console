"""Fake observations only: no OSC service, camera driver or CAN writes."""
import copy
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from supervisor.inference_alignment import SampleHistory, AlignmentPending, nearest_frames, next_grid_anchor
from supervisor.inference_rtc import ObservationAssembler, Timeline, Target, PERIOD_NS
from supervisor.inference_rtc_timing import RTCTiming


class AlignmentTests(unittest.TestCase):
    def test_joint_interpolation_is_bracketed_and_never_refreshes_raw_times(self):
        h = SampleHistory()
        h.append('s', 340_000_000, [0.]*7)
        h.append('s', 360_000_000, [2.]*7)
        value, evidence = h.at('s', 350_000_000)
        np.testing.assert_array_equal(value, [1.]*7)
        self.assertEqual(evidence['source_perf_ns'], [340_000_000,360_000_000])
        self.assertEqual(evidence['fraction'], .5)
        for anchor in (330_000_000,370_000_000):
            with self.assertRaises(AlignmentPending): h.at('s', anchor)
        with self.assertRaises(AlignmentPending): h.at('new',350_000_000)
        h.append('new', 380_000_000, [3.]*7)
        with self.assertRaises(AlignmentPending): h.at('s',350_000_000)

    def test_duplicate_reversed_nonfinite_and_large_gaps(self):
        h = SampleHistory()
        h.append(1,100,[1.]); h.append(1,100,[2.]); h.append(1,90,[3.])
        h.append(1,110,[float('nan')])
        self.assertEqual(h.at(1,100)[0][0],1.)
        h.append(1,200_000_000,[2.])
        with self.assertRaisesRegex(RuntimeError,'gap'): h.at(1,100_000_000)

    def test_real_camera_selection_keeps_original_timestamps(self):
        r = SimpleNamespace(lock=threading.Lock(),frames={},frame_times_ns={},frame_history={
            k:[(340_000_000,np.array([1])),(373_333_333,np.array([2]))]
            for k in ('external','wrist')})
        self.assertEqual(nearest_frames(r,350_000_000)[1],[340_000_000]*2)
        self.assertEqual(nearest_frames(r,358_000_000)[1],[373_333_333]*2)
        r.frame_history['wrist']=[]
        with self.assertRaises(AlignmentPending): nearest_frames(r,350_000_000)

    def test_parameter_driven_grid_and_exact_corner_prefix_no_double_resampling(self):
        origin = 1_000_000_007
        for d,e in ((3,5),(5,7),(6,6),(8,8)):
            timing = RTCTiming(d,e)
            anchor = next_grid_anchor(origin,origin+183_412_000,PERIOD_NS)
            for iteration in range(100):
                self.assertEqual((anchor-origin)%PERIOD_NS,0)
                self.assertEqual(anchor,origin+(4+iteration*e)*PERIOD_NS)
                timeline = Timeline(1,origin)
                # A sharp corner, for which off-grid round-trip interpolation
                # changes the geometry even if prediction exactly fits prefix.
                start=(anchor-origin)//PERIOD_NS
                timeline.targets={i:Target((float(i%3),0,.3),(0,0,0,1),.5)
                                  for i in range(start,start+20)}
                ctx=timeline.context(start,d,iteration+1,observation_ns=anchor,
                                     now_ns=anchor,execute_horizon=timing.execute_horizon)
                for j,row in enumerate(ctx['absolute_prefix']):
                    self.assertEqual(row,timeline.targets[start+j].absolute_row())
                anchor += timing.execute_horizon*PERIOD_NS

    def hardware_assembler(self):
        anchor=time.perf_counter_ns()-20_000_000
        key=('s','c','hardware')
        joints=SampleHistory();grip=SampleHistory()
        for stamp,value in ((anchor-10_000_000,0.),(anchor+10_000_000,2.)):
            joints.append(key,stamp,[value]*7)
            grip.append('s',stamp,[value*.01])
        frames=SimpleNamespace(lock=threading.Lock(),frames={},frame_times_ns={},
            frame_history={k:[(anchor-5_000_000,np.zeros((2,2,3),np.uint8))]
                           for k in ('external','wrist')})
        seen=[]
        def fk(q):
            seen.append(q.copy())
            return {'position_m':[float(q[0]),0,.3],'orientation_xyzw':[0,0,0,1]}
        a=SimpleNamespace(_tcp_feed=SimpleNamespace(history=joints),
            _observation_source=SimpleNamespace(pose_for_joints=fk),
            _measured_gripper_cache=SimpleNamespace(history=grip),camera_resource=frames,
            _event=lambda *args,**kwargs:None,
            _observation=lambda osc,*frames: {'aligned':copy.deepcopy(osc['inference_observation'])})
        assembler=ObservationAssembler(a)
        assembler.read=lambda: ({}, {}, {'session':dict(zip(('id','client_id','execution_mode'),key)),
            'inference_observation':{'perf_ns':anchor+10_000_000}},anchor+10_000_000)
        return anchor,a,assembler,seen

    def test_hardware_uses_fk_of_interpolated_joints_and_measured_gripper(self):
        anchor,a,assembler,seen=self.hardware_assembler()
        obs,pose,osc,stamp=assembler.read_aligned(anchor,threading.Event())
        np.testing.assert_array_equal(seen[0],[1.]*7)
        self.assertEqual(pose['position_m'][0],1.)
        self.assertAlmostEqual(obs['aligned']['gripper_width_m'],.01)
        self.assertEqual(stamp,anchor)
        evidence=osc['rtc_observation_times']
        self.assertEqual(evidence['image_offset_ms'],[-5.,-5.])
        self.assertEqual(evidence['raw_latest_tcp_perf_ns'],anchor+10_000_000)

    def test_missing_history_stop_and_session_change_do_not_fabricate_state(self):
        anchor,a,assembler,seen=self.hardware_assembler()
        cancelled=threading.Event();cancelled.set()
        self.assertIsNone(assembler.read_aligned(anchor,cancelled))
        self.assertEqual(seen,[])
        # Already expired history cannot be replaced by the latest pose.
        a._tcp_feed.history=SampleHistory()
        with self.assertRaisesRegex(RuntimeError,'missing bracketing'):
            assembler.read_aligned(anchor-200_000_000,threading.Event())


if __name__=='__main__':unittest.main()
