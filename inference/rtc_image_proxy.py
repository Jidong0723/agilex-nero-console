"""AutoDL image transport only; JPEG95 negotiated explicitly, legacy8->7."""
import asyncio
import logging
import traceback
import cv2
import msgpack
import numpy as np
import websockets
from websockets.asyncio.server import serve

def pack(value):
    if isinstance(value,np.ndarray):return {b'__ndarray__':True,b'data':value.tobytes(),b'dtype':value.dtype.str,b'shape':value.shape}
    if isinstance(value,np.generic):return {b'__npgeneric__':True,b'data':value.item(),b'dtype':value.dtype.str}
    raise TypeError(type(value).__name__)
def unpack(value):
    if b'__ndarray__' in value:return np.ndarray(buffer=value[b'data'],dtype=np.dtype(value[b'dtype']),shape=value[b'shape'])
    if b'__npgeneric__' in value:return np.dtype(value[b'dtype']).type(value[b'data'])
    return value
def adapt(raw, previous=None):
    previous = {} if previous is None else previous
    obs=msgpack.unpackb(raw,object_hook=unpack)
    state=np.asarray(obs['observation/state'],np.float32)
    if state.shape==(8,) and np.isclose(state[7],-state[6],rtol=0,atol=1e-4):state=state[:7].copy()
    if state.shape!=(7,) or not np.isfinite(state).all():raise ValueError('invalid NERO state')
    obs['observation/state']=state
    for key in ('observation/image','observation/wrist_image'):
        item=obs[key]
        if isinstance(item,dict) and item.get('encoding')=='repeat_rgb_v1':
            if key not in previous:raise ValueError('RGB repeat without connection reference')
            obs[key]=previous[key].copy()
        elif isinstance(item,dict) and item.get('encoding') in ('png_rgb_v1','png_xor_rgb_v1','jpeg_rgb_v1'):
            bgr=cv2.imdecode(np.frombuffer(item['data'],np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:raise ValueError('invalid encoded RGB payload')
            obs[key]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            if item['encoding']=='png_xor_rgb_v1':
                if key not in previous or previous[key].shape!=obs[key].shape:raise ValueError('RGB delta reference missing')
                obs[key]=np.bitwise_xor(obs[key],previous[key])
        previous[key]=np.asarray(obs[key]).copy()
    return msgpack.Packer(default=pack).pack(obs)
async def handler(downstream):
    try:
        async with websockets.connect('ws://127.0.0.1:8001',compression=None,max_size=None,open_timeout=120) as upstream:
            metadata=msgpack.unpackb(await upstream.recv(),object_hook=unpack)
            metadata['nero_image_transport']='jpeg95_rgb_v1'
            await downstream.send(msgpack.Packer(default=pack).pack(metadata))
            previous={}
            while True:
                await upstream.send(adapt(await downstream.recv(),previous))
                await downstream.send(await upstream.recv())
    except websockets.ConnectionClosed:return
    except Exception:
        logging.exception('RTC image proxy failure')
        try:await downstream.send(traceback.format_exc());await downstream.close(code=1011)
        except Exception:pass
async def main():
    async with serve(handler,'0.0.0.0',8000,compression=None,max_size=None):await asyncio.Future()
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
