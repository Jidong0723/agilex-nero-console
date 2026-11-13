import unittest,tempfile,types,threading,json,time,io,zipfile
from pathlib import Path
from supervisor.recap_extension import Collection,guard

class Recorder:
 def __init__(self,*a,**k):self.root=Path(a[2]);self.active=False
 def start(self,body):self.active=True;self.root.mkdir(parents=True);(self.root/'raw.json').write_text('{}');return {'episode_dir':str(self.root)}
 def stop(self,*a):self.active=False;return {'episode_dir':str(self.root)}

def fixture(root):
 a=types.SimpleNamespace();a.calls=[]
 a.session={'id':'s','client_id':'c','state':'ACTIVE','execution_mode':'shadow'}
 a.pi={'execution_enabled':False,'policy_metadata':{'state_dim':7,'valid_action_dim':7,'action_horizon':20},'gripper_transport':{},'recovery':{}}
 def start(s,c):a.calls.append('model-start');a.pi.update(execution_enabled=True,session_id=s,client_id=c)
 def stop(reason):a.calls.append('model-stop');a.pi['execution_enabled']=False
 def update(body):a.calls.append('prompt');a.pi.update(model_state='CONNECTED',protocol_compatibility={'status':'compatible'});return {}
 a.pi05=types.SimpleNamespace(snapshot=lambda:dict(a.pi),start=start,stop=stop,update_config=update,export_inference_log=lambda:('log.zip',b'complete-original-model-log'))
 a.osc=types.SimpleNamespace(fast_state=lambda:{'session':dict(a.session),'diagnostics':{'trajectory_state':'HOLD_READY'}},input_context=lambda:{'connected':True},bind_source=lambda *args:a.calls.append('bind-'+args[-1]))
 a.pico=types.SimpleNamespace(snapshot=lambda:{})
 a.cameras=types.SimpleNamespace(snapshot=lambda:{'ready':True});a.dataset=types.SimpleNamespace(state=lambda:{'recording':False})
 a.pi05_start=lambda s,c:(a.osc.bind_source(s,c,'pi05'),start(s,c))
 g=types.SimpleNamespace(session=None)
 def human(s,c):a.calls.append('pico-start');g.session=s
 def invalidate():a.calls.append('pico-stop');g.session=None
 g.start_session=human;g.invalidate=invalidate;g.status=lambda:{'session_id':g.session}
 m=Collection(a,g,root,Recorder)
 return a,g,m

class Tests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.a,self.g,self.m=fixture(self.tmp.name)
 def tearDown(self):
  if self.m.state()['recording']:self.m.finish('c','aborted')
  self.tmp.cleanup()
 def begin(self,purpose='pilot'):return self.m.start(dict(session_id='s',client_id='c',task='move cube',scene='scene01',purpose=purpose))
 def test_construct_and_state_are_inert(self):
  self.m.state();self.assertEqual(self.a.calls,[]);self.assertEqual(list(Path(self.tmp.name).iterdir()),[])
 def test_multiple_handover_fresh_start_and_full_model_logs(self):
  self.begin()
  for _ in range(2):
   self.m.switch('c',True);self.assertFalse(self.a.pi['execution_enabled']);self.assertEqual(self.g.session,'s')
   self.m.switch('c',False);self.assertTrue(self.a.pi['execution_enabled']);self.assertIsNone(self.g.session)
  self.m.finish('c','success');self.assertEqual(self.m.meta['interventions'],2)
  self.assertFalse(self.m.meta['autonomous_success']);self.assertTrue(self.m.meta['assisted_success'])
  self.assertEqual(len(self.m.meta['model_logs']),3)
  self.assertEqual(self.a.calls.count('model-start'),3)
  name,payload=self.m.export()
  with zipfile.ZipFile(io.BytesIO(payload)) as z:self.assertIn('episode.json',z.namelist());self.assertIn('raw/raw.json',z.namelist())
 def test_model_stops_before_pico_binds(self):
  self.begin();self.m.switch('c',True)
  self.assertLess(self.a.calls.index('model-stop'),self.a.calls.index('pico-start'))
 def test_waits_for_prompt_reconnect_before_starting(self):
  def update(body):
   self.a.pi.update(model_state='CONNECTING',protocol_compatibility={'status':'pending'})
   def connected():
    self.a.calls.append('handshake')
    self.a.pi.update(model_state='CONNECTED',protocol_compatibility={'status':'compatible'})
   threading.Timer(.08,connected).start()
  self.a.pi05.update_config=update
  self.begin()
  self.assertLess(self.a.calls.index('handshake'),self.a.calls.index('model-start'))
 def test_hardware_priming_does_not_end_collection(self):
  self.a.pi05.start=lambda s,c:self.a.pi.update(priming=True,execution_enabled=False,session_id=s,client_id=c)
  self.assertEqual(self.begin()['state'],'PRIMING')
  time.sleep(.45)
  self.assertTrue(self.m.state()['recording'])
  self.a.pi.update(priming=False,execution_enabled=False,last_error='Real priming failure')
  deadline=time.monotonic()+2
  while self.m.state()['recording'] and time.monotonic()<deadline:time.sleep(.02)
  self.assertEqual(self.m.state()['error'],'Real priming failure')
 def test_invalid_owner_never_changes_control(self):
  self.begin();calls=list(self.a.calls)
  with self.assertRaises(PermissionError):self.m.switch('other',True)
  self.assertEqual(self.a.calls,calls)
 def test_invalid_metadata_never_starts(self):
  self.a.pi['policy_metadata']['action_horizon']=10
  with self.assertRaises(RuntimeError):self.begin()
  self.assertEqual(self.a.calls,[])
 def test_demo_does_not_start_model(self):
  self.begin('demo');self.assertNotIn('model-start',self.a.calls);self.assertEqual(self.m.phase,'HUMAN')
 def test_abort_is_not_failure(self):
  self.begin();self.m.finish('c','aborted');self.assertEqual(self.m.meta['outcome'],'aborted')
 def test_start_cancelled_does_not_advance_to_model(self):
  original=self.a.pi05.update_config
  def cancel(body):original(body)
  self.a.pi05.update_config=cancel
  self.begin();self.m.cancel.set()
  with self.assertRaises(RuntimeError):self.m.switch('c',True)
  self.assertNotIn('pico-start',self.a.calls)
 def test_pending_row_wait_cancel_does_not_grant_pico(self):
  self.begin();self.a.pi05._row_lock=threading.Lock();self.a.pi05._row_lock.acquire()
  threading.Timer(.05,self.m.cancel.set).start()
  with self.assertRaises(RuntimeError):self.m.switch('c',True)
  self.a.pi05._row_lock.release();self.assertNotIn('pico-start',self.a.calls)
 def test_session_loss_does_not_stop_foreign_controller(self):
  self.begin();self.a.session['id']='foreign';self.a.pi.update(session_id='foreign',client_id='other')
  self.g.session='foreign';self.m.finish('c','aborted')
  self.assertNotIn('model-stop',self.a.calls);self.assertNotIn('pico-stop',self.a.calls)
 def test_route_guard_only_when_opted_in(self):
  runtime=types.SimpleNamespace();guard(runtime,'/api/pi05/start')
  runtime.recap_extension=self.m;guard(runtime,'/api/pi05/start');self.begin()
  with self.assertRaises(RuntimeError):guard(runtime,'/api/pi05/start')
  guard(runtime,'/api/osc/session/stop');guard(runtime,'/api/emergency/damping')
  guard(runtime,'/api/osc/command',{'type':'hold'})

if __name__=='__main__':unittest.main()
