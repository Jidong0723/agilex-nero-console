import unittest
import copy
import ast
from pathlib import Path
import numpy as np
import time
from types import SimpleNamespace
from supervisor.inference_rtc_timing import RTCTiming, preserve_delay_prefix
from supervisor.inference_rtc import Timeline, Target, PERIOD_NS
from supervisor.inference_config import project, prepare_update
from inference.rtc.codec import validate_context, echo_context, pack_normalized_prefix, guidance_bounds


class FixedTimingTests(unittest.TestCase):
    def test_bootstrap_attends_real_full_hold_but_replan_keeps_official_overlap(self):
        timeline=Timeline(1,1_000_000_000)
        timeline.seed([Target((0,0,.3),(0,0,0,1),.5)]*20)
        replan=timeline.context(0,8,1,observation_ns=1_000_000_000,execute_horizon=10)
        hold=dict(replan,phase='bootstrap_hold',prefix_attention_horizon=20)
        self.assertEqual(validate_context(hold),(8,20))
        self.assertEqual(validate_context(replan),(8,10))
        for context in (dict(hold,prefix_attention_horizon=10),dict(replan,prefix_attention_horizon=20)):
            with self.assertRaisesRegex(ValueError,'prefix_attention_horizon'):validate_context(context)

    def test_bootstrap_preserves_real_hold_not_approximate_guided_predictions(self):
        old=['HOLD']*20; predicted=list(range(20))
        self.assertEqual(preserve_delay_prefix(old,predicted,8),['HOLD']*8+list(range(8,20)))
        self.assertEqual(predicted,list(range(20)))
        self.assertEqual(preserve_delay_prefix(old,predicted,0),predicted)

    def test_real_server_infer_body_uses_distinct_prefix_and_overlap_lengths(self):
        root=Path(__file__).parents[1]/'inference/rtc'
        tree=ast.parse((root/'serve.py').read_text())
        klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RtcPolicy')
        method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='_infer')
        weight=next(n for n in ast.parse((root/'sampling.py').read_text()).body
                    if isinstance(n,ast.FunctionDef) and n.name=='prefix_weights')
        fakejax=SimpleNamespace(tree=SimpleNamespace(map=lambda f,d:{k:f(v) for k,v in d.items()}),
            random=SimpleNamespace(split=lambda x:(x,x)),block_until_ready=lambda x:x)
        scope=dict(np=np,jnp=np,jax=fakejax,time=time,HORIZON=20,validate_context=validate_context,
            # Coordinate conversion/GPU are separate integration checks; here
            # exercise the actual server's complete array assembly/diagnostics.
            encode_prefix=lambda rows,state:np.zeros((len(rows),7),np.float32),
            echo_context=echo_context,pack_normalized_prefix=pack_normalized_prefix,guidance_bounds=guidance_bounds,
            prefix_weights=lambda start,end,schedule='exp': np.ones(20,np.float32),
            model_module=SimpleNamespace(Observation=SimpleNamespace(from_dict=lambda d:d)))
        exec(compile(ast.Module(body=[weight,method],type_ignores=[]),'actual_cloud_infer','exec'),scope)
        owner=SimpleNamespace(normalize=lambda d:d,steps=10,cap=5.,
            policy=SimpleNamespace(_rng=0,_input_transform=lambda d:{'state':d['observation/state']},
                _output_transform=lambda d:{'actions':d['actions'][:,:7]}),
            rtc_sample=lambda *a,**kw:np.zeros((1,20,32),np.float32))
        timeline=Timeline(1,1_000_000_000)
        timeline.seed([Target((0,0,.3),(0,0,0,1),.5)]*20)
        context=timeline.context(0,8,1,observation_ns=1_000_000_000,execute_horizon=10)
        obs={'observation/state':np.array([0,0,.3,0,0,0,.5],np.float32)}
        for phase in ('replan','bootstrap_hold'):
            request=dict(obs,rtc_context=dict(context,phase=phase,
                prefix_attention_horizon=20 if phase=='bootstrap_hold' else 10))
            response=scope['_infer'](owner,request)
            self.assertEqual(response['actions'].shape,(20,7))
            self.assertEqual(response['rtc']['valid_prefix_steps'],20 if phase=='bootstrap_hold' else 10)
            self.assertTrue(np.isfinite(response['rtc']['prefix_position_rmse_m']))
            self.assertEqual(response['rtc']['sampler_config']['num_steps'], 10)
            self.assertEqual(len(response['rtc']['prefix_residuals']), 20 if phase=='bootstrap_hold' else 10)
            self.assertEqual(response['rtc']['prefix_residuals'][0]['region'], 'hard')
        owner.requires_guided_execution=True
        unsafe=timeline.context(0,7,2,observation_ns=1_000_000_000,execute_horizon=8)
        with self.assertRaisesRegex(ValueError,r'2\*execute_horizon'):
            scope['_infer'](owner,dict(obs,rtc_context=unsafe))
        safe=timeline.context(0,6,3,observation_ns=1_000_000_000,execute_horizon=6)
        self.assertEqual(scope['_infer'](owner,dict(obs,rtc_context=safe))['actions'].shape,(20,7))
        owner.requires_guided_execution=False
        official=timeline.context(0,8,4,observation_ns=1_000_000_000,execute_horizon=8)
        self.assertEqual(scope['_infer'](owner,dict(obs,rtc_context=official))['actions'].shape,(20,7))
        captured={}
        def capture(*args,**kwargs):
            captured['weights']=np.asarray(args[3]);return np.zeros((1,20,32),np.float32)
        owner.rtc_sample=capture
        off_grid=timeline.context(8,8,5,observation_ns=1_425_000_000,execute_horizon=8)
        scope['_infer'](owner,dict(obs,rtc_context=off_grid))
        self.assertAlmostEqual(captured['weights'][0,8,0],1.)
        self.assertGreater(captured['weights'][0,9,0],.48)
        np.testing.assert_array_equal(captured['weights'][0,11:],np.zeros((9,32)))

    def test_full_physical_prefix_does_not_use_short_guidance_length(self):
        for length in (10,13,20):
            values=np.arange(length*7,dtype=np.float32).reshape(length,7)
            packed=pack_normalized_prefix(values)
            self.assertEqual(packed.shape,(1,20,32))
            np.testing.assert_array_equal(packed[0,:length,:7],values)
            np.testing.assert_array_equal(packed[0,:,7:],np.zeros((20,25)))
            np.testing.assert_array_equal(packed[0,length:,:7],np.zeros((20-length,7)))

    def test_actual_cloud_prefix_weights_match_upstream_schedule(self):
        # Load the real function without importing GPU-only OpenPI/JAX modules.
        source=Path(__file__).parents[1]/'inference/rtc/sampling.py'
        node=next(n for n in ast.parse(source.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name=='prefix_weights')
        scope={'jnp':np}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        weights=scope['prefix_weights'](2,6,10)
        linear=np.array([1,1,.8,.6,.4,.2,0,0,0,0])
        expected=linear*np.expm1(linear)/(np.e-1)
        np.testing.assert_allclose(weights,expected,atol=1e-7)
        np.testing.assert_array_equal(scope['prefix_weights'](0,0),np.zeros(20))
        np.testing.assert_array_equal(scope['prefix_weights'](9,20,schedule='ones'),np.ones(20))
        np.testing.assert_allclose(scope['prefix_weights'](2,6,10,schedule='linear'),linear,atol=1e-7)
        np.testing.assert_array_equal(scope['prefix_weights'](2,6,10,schedule='zeros'),np.array([1,1,0,0,0,0,0,0,0,0]))

    def test_configuration_validation_and_migration(self):
        config = project({'execution': {'rtc_min_replan_steps': 9}})
        self.assertNotIn('rtc_min_replan_steps', config['execution'])
        self.assertEqual(RTCTiming.from_execution(config['execution']), RTCTiming(6, 6))
        for d, e in [(False, 8), (-1, 8), (9, 8), (10, 11), (0, 0), (1, 20), (1.0, 8)]:
            with self.subTest(d=d,e=e), self.assertRaises(ValueError):
                RTCTiming(d, e)
        for d,e in [(0,1),(2,6),(8,10),(10,10)]:
            self.assertEqual(RTCTiming(d,e).prefix_attention_horizon,20-e)
        with self.assertRaisesRegex(ValueError,'retired'):
            prepare_update(config, {'execution': {'rtc_min_replan_steps': 8}})

    def test_official_splice_and_shift_on_absolute_timeline(self):
        old = [Target((i*.0001,0,.3),(0,0,0,1),.5) for i in range(20)]
        new = [Target((.01+i*.0001,0,.3),(0,0,0,1),.5) for i in range(20)]
        origin=1_000_000_000
        for delay, horizon in [(0,1),(2,6),(8,8),(8,10)]:
            timeline=Timeline(1,origin);timeline.seed(old)
            context=timeline.context(0,delay,1,observation_ns=origin,now_ns=origin,execute_horizon=horizon)
            self.assertEqual(validate_context(context),(delay,20-horizon))
            # Same ideal fixed-delay sequence as upstream eval_flow.py.
            timeline.targets=timeline.merged(context,new,now_ns=origin-1)
            executed=[]
            for i in range(horizon):
                tick,target=timeline.commit_next();executed.append(target);timeline.complete(tick)
            self.assertEqual(executed,old[:delay]+new[delay:horizon])
            self.assertEqual(timeline.targets[horizon],new[horizon])
            self.assertEqual(timeline.next_tick,horizon)
            self.assertEqual(echo_context(context)['execute_horizon'],horizon)

    def test_delay_is_not_shifted_by_observation_age_or_request_clock(self):
        origin=1_000_000_000; timeline=Timeline(1,origin)
        timeline.seed([Target((0,0,.3),(0,0,0,1),.5)]*20)
        context=timeline.context(0,8,1,observation_ns=origin,now_ns=origin+40_000_000,execute_horizon=10)
        self.assertEqual(context['freeze_until_perf_ns'],origin+8*PERIOD_NS)
        self.assertEqual(context['prefix_attention_horizon'],10)
        corrupt=copy.deepcopy(context);corrupt['prefix_attention_horizon']=11
        with self.assertRaisesRegex(ValueError,'H - execute_horizon'):validate_context(corrupt)

    def test_expired_response_never_retimes_row_zero(self):
        origin=1_000_000_000;timeline=Timeline(1,origin)
        old=[Target((i*.001,0,.3),(0,0,0,1),.5) for i in range(20)]
        timeline.seed(old)
        context=timeline.context(0,2,1,observation_ns=origin,execute_horizon=6)
        merged=timeline.merged(context,old,now_ns=origin+4*PERIOD_NS)
        self.assertEqual(merged[5],old[5])
        self.assertNotEqual(merged[5],old[0])

    def test_late_guided_response_keeps_committed_targets_and_original_time(self):
        origin = 1_000_000_000
        timeline = Timeline(1, origin)
        old = [Target((i*.0002, 0, .3), (0,0,0,1), .5) for i in range(20)]
        timeline.seed(old)
        context = timeline.context(0, 6, 1, observation_ns=origin, execute_horizon=6)
        timeline.committed_until = 8
        new = [Target((i*.0002+.0001, 0, .3), (0,0,0,1), .5) for i in range(20)]
        metadata = dict(prefix_attention_schedule='ones', rtc_requires_execution_within_prefix=True)
        for schedule in ('ones', 'linear', 'exp'):
            metadata['prefix_attention_schedule'] = schedule
            candidate, evidence = timeline.merge_response(context, new, origin+301_000_000, metadata)
            self.assertEqual(evidence['replacement_tick'], 8)
            self.assertEqual(evidence['response_lateness_ms'], 1.)
            self.assertEqual(candidate[7], old[7])
            self.assertEqual(candidate[8], new[8])
            self.assertAlmostEqual(evidence['same_time_position_gap_m'], .0001)

    def test_late_response_without_guidance_or_safe_seam_still_rejects(self):
        origin = 1_000_000_000
        timeline = Timeline(1, origin)
        old = [Target((0,0,.3),(0,0,0,1),.5)]*20
        timeline.seed(old)
        context = timeline.context(0,6,1,observation_ns=origin,execute_horizon=6)
        metadata = dict(prefix_attention_schedule='ones', rtc_requires_execution_within_prefix=True)
        with self.assertRaisesRegex(RuntimeError, 'supported future'):
            timeline.merge_response(context,old,origin+301_000_000,{})
        unsafe = [Target((.02,0,.3),(0,0,0,1),.5)]*20
        for schedule in ('ones', 'linear', 'exp'):
            metadata['prefix_attention_schedule'] = schedule
            with self.assertRaisesRegex(RuntimeError, 'safely match'):
                timeline.merge_response(context,unsafe,origin+301_000_000,metadata)
            with self.assertRaisesRegex(RuntimeError, 'guided old-plan'):
                timeline.merge_response(context,old,origin+701_000_000,metadata)
        self.assertEqual(timeline.targets[0], old[0])

    def test_exp_eight_eight_allows_free_tail_but_late_cut_needs_guided_reference(self):
        origin=1_000_000_000;timeline=Timeline(1,origin)
        old=[Target((i*.0001,0,.3),(0,0,0,1),.5) for i in range(20)]
        timeline.seed(old)
        context=timeline.context(0,8,1,observation_ns=origin,execute_horizon=8)
        self.assertEqual(validate_context(context),(8,12))
        metadata=dict(prefix_attention_schedule='exp',rtc_requires_execution_within_prefix=False,
                      rtc_late_response_policy='guided_same_time_continuity_v1')
        # Rows12:16 are normal unconstrained future generation, not discarded.
        candidate,evidence=timeline.merge_response(context,old,origin+390_000_000,metadata)
        self.assertEqual(candidate[15],old[15]);self.assertEqual(evidence['replacement_tick'],8)
        candidate,evidence=timeline.merge_response(context,old,origin+401_000_000,metadata)
        self.assertEqual(evidence['replacement_tick'],9)
        with self.assertRaisesRegex(RuntimeError,'guided old-plan'):
            timeline.merge_response(context,old,origin+551_000_000,metadata)
        unsafe=[Target((.02,0,.3),(0,0,0,1),.5)]*20
        with self.assertRaisesRegex(RuntimeError,'safely match'):
            timeline.merge_response(context,unsafe,origin+401_000_000,metadata)
        with self.assertRaisesRegex(RuntimeError,'safely match'):
            timeline.merge_response(context,unsafe,origin+220_000_000,metadata)

    def test_off_grid_mask_keeps_nominal_overlap_and_freezes_real_rows(self):
        timeline=Timeline(1,1_000_000_000)
        timeline.seed([Target((0,0,.3),(0,0,0,1),.5)]*20)
        c=timeline.context(8,8,1,observation_ns=1_425_000_000,execute_horizon=8)
        self.assertEqual(len(c['absolute_prefix']),11)
        self.assertEqual(guidance_bounds(c,'legacy_compressed'),(8,11,11))
        self.assertEqual(guidance_bounds(c,'nominal'),(8,12,11))
        self.assertEqual(guidance_bounds(c,'timestamp_frozen'),(9,12,11))
        # Timestamp mapping is exact on the official integer-step grid.
        c=timeline.context(8,8,2,observation_ns=1_400_000_000,execute_horizon=8)
        self.assertEqual(guidance_bounds(c),(8,12,12))


if __name__=='__main__':unittest.main()
