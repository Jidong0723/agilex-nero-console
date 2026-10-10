"""Offline same-noise image transport audit; no robot or control imports."""
import json
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

def audit(policy, output):
    root=Path('/root/autodl-tmp/nero_deploy/validation/nero_20sample_source')
    episodes=[json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
    episodes=[ep for ep in episodes if ep['episode_index'] < 20]
    if len(episodes)!=20:raise RuntimeError('Expected the prepared 20 teacher episodes')
    rows=[]
    def image(path, frame):
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,bgr=cap.read();cap.release()
        if not ok:raise RuntimeError(f'Missing audit frame {path}:{frame}')
        h,w=bgr.shape[:2];scale=min(224/w,224/h)
        bgr=cv2.resize(bgr,(round(w*scale),round(h*scale)))
        canvas=np.zeros((224,224,3),np.uint8);y=(224-bgr.shape[0])//2;x=(224-bgr.shape[1])//2
        canvas[y:y+bgr.shape[0],x:x+bgr.shape[1]]=bgr
        return cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)
    for ep in episodes:
        index=ep['episode_index'];frame=round((ep['length']-11)*(.12+.76*(index%5)/4))
        table=pq.read_table(root/f'data/chunk-000/episode_{index:06d}.parquet')
        state=np.asarray(table['observation.state'][frame].as_py(),np.float32)
        obs={'observation/state':state,'prompt':ep['tasks'][0]}
        compressed=dict(obs);sizes=[];pixel=[]
        for key,name in [('observation/image','base_0_rgb'),('observation/wrist_image','left_wrist_0_rgb')]:
            rgb=image(root/f'videos/chunk-000/observation.images.{name}/episode_{index:06d}.mp4',frame)
            ok,data=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok:raise RuntimeError('JPEG95 audit encoding failed')
            decoded=cv2.cvtColor(cv2.imdecode(data,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            obs[key]=rgb;compressed[key]=decoded;sizes.append(data.nbytes)
            pixel.append(float(np.abs(rgb.astype(float)-decoded).mean()))
        noise=np.random.default_rng(20261006+index).normal(size=(1,10,32)).astype(np.float32)
        raw=np.asarray(policy.infer(obs,noise=noise)['actions'])
        jpeg=np.asarray(policy.infer(compressed,noise=noise)['actions'])
        delta=jpeg-raw
        rot=(Rotation.from_rotvec(jpeg[:,3:6])*Rotation.from_rotvec(raw[:,3:6]).inv()).magnitude()*180/np.pi
        rows.append(dict(episode=index,frame=frame,payload_image_bytes=sum(sizes),pixel_mae=float(np.mean(pixel)),
                         position_difference_mm=(np.linalg.norm(delta[:,:3],axis=1)*1000).tolist(),
                         rotation_difference_deg=rot.tolist(),gripper_difference_mm=(np.abs(delta[:,6])*.095*1000).tolist()))
    def stats(values):
        a=np.asarray(values);return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))
    report=dict(no_robot=True,quality=95,same_noise=True,episodes=len(rows),actions=len(rows)*10,
                position_difference_mm=stats([v for r in rows for v in r['position_difference_mm']]),
                rotation_difference_deg=stats([v for r in rows for v in r['rotation_difference_deg']]),
                gripper_difference_mm=stats([v for r in rows for v in r['gripper_difference_mm']]),
                image_bytes=stats([r['payload_image_bytes'] for r in rows]),rows=rows,
                caveat='JPEG versus original image prediction perturbation, not task success or ground truth error.')
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
