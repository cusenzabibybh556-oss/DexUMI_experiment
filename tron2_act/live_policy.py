"""Frozen prior-only ACT inference, shared by offline replay and live workers."""
from pathlib import Path
import hashlib
import json
import queue
import time

import cv2
import numpy as np

from .config import CONTRACT,Config


class ForegroundUnavailable(RuntimeError):
    """One observation cannot produce the configured ACT visual input."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_geometry(fg):
    result={}
    for key in ('hand','objects'):
        y,x=np.where(fg[key])
        result[key]={
            'centroid':[float(x.mean()),float(y.mean())] if len(x) else None,
            'bbox':[int(x.min()),int(y.min()),int(x.max()),int(y.max())] if len(x) else None,
        }
    return result


class LivePolicy:
    def __init__(self,config,root,device='cuda'):
        import torch
        from .model import ACT
        from tron2_vision.model import ForegroundPredictor
        self.torch=torch;self.device=device
        torch.set_num_threads(4)
        checkpoint=Path(root)/config['checkpoint'];foreground=Path(root)/config['foreground_checkpoint']
        if digest(checkpoint)!=config['checkpoint_sha256'] or digest(foreground)!=config['foreground_sha256']:
            raise ValueError('Policy/foreground weight hash differs from deployment config')
        bundle=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if bundle['contract']!=CONTRACT:raise ValueError('Unsupported ACT action contract')
        self.step=int(bundle['step'])
        self.config=Config(**bundle['config']).validate()
        if self.config.action_hz!=20 or self.config.history_steps!=2:
            raise ValueError('Live adapter requires 20Hz and two wrist history steps')
        self.stats={k:np.asarray(bundle['stats'][k],np.float32) for k in ['state_mean','state_std','action_mean','action_std']}
        if any(not np.isfinite(x).all() for x in self.stats.values()) or np.any(self.stats['state_std']<=0) or np.any(self.stats['action_std']<=0):
            raise ValueError('Invalid train-only normalization statistics')
        self.model=ACT(self.config,load_pretrained=False).to(device).eval()
        self.model.load_state_dict(bundle['model'])
        self.foreground=ForegroundPredictor(foreground,device=device)
        self.runtime_config=config;self.alignment=None;self.source_info=None
        self.target_camera=None
        if config['align_camera_intrinsics']:
            from tools.preview_tron2_top_camera import read_reference
            self.target_camera,_=read_reference(Path(config['reference_hdf5']),0)
        with torch.inference_mode():
            self.model(torch.zeros(1,3,*self.config.image_size,device=device),
                       torch.zeros(1,self.config.state_dim,device=device))
            self.foreground.predict(np.zeros((*self.config.image_size,3),np.uint8))
        if device.startswith('cuda'):torch.cuda.synchronize()

    def prepare_vision(self,rgb,camera_info=None,already_aligned=False):
        """Same aligned pixels and semantic masks for the camera check and ACT."""
        if self.runtime_config['align_camera_intrinsics'] and not already_aligned:
            from tools.preview_tron2_top_camera import CameraModel,CameraViewAlignment,optical_rotation
            if camera_info is None:
                raise ValueError('Top CameraInfo required for optical alignment')
            if self.source_info is not None and camera_info!=self.source_info:
                raise RuntimeError('Top camera intrinsics changed during deployment')
            if self.alignment is None:
                self.source_info=camera_info
                rotation=optical_rotation(self.runtime_config.get('virtual_look_down_deg',0.),
                                          self.runtime_config.get('virtual_look_right_deg',0.))
                alignment=CameraViewAlignment(CameraModel(**camera_info),self.target_camera,rotation)
                if alignment.valid_fraction<.95:raise RuntimeError('Training camera view not covered by Top camera')
                self.alignment=alignment
            rgb=self.alignment.apply(rgb)
        height,width=self.config.image_size
        rgb=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA)
        fg=self.foreground.predict(rgb)
        return rgb,fg

    @staticmethod
    def check_masks(fg):
        counts=[int(fg[k].sum()) for k in ['hand','objects']]
        if counts[0]<100 or counts[1]<50:
            raise ForegroundUnavailable('Foreground mask insufficient: '+str(counts))
        return counts

    def predict(self,rgb,state,camera_info=None,already_aligned=False,return_aligned=False):
        torch=self.torch;tick=time.perf_counter()
        aligned,fg=self.prepare_vision(rgb,camera_info,already_aligned)
        counts=self.check_masks(fg)
        state=np.asarray(state,np.float32)
        if state.shape!=(self.config.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f'Invalid aligned {self.config.state_dim}D state')
        normalized=(state-self.stats['state_mean'])/self.stats['state_std']
        image=torch.from_numpy(np.ascontiguousarray(fg['rgb'].transpose(2,0,1))).to(self.device).float()[None]/255
        with torch.inference_mode():
            predicted,_,_=self.model(image,torch.from_numpy(normalized).to(self.device)[None])
        action=predicted[0].cpu().numpy()*self.stats['action_std']+self.stats['action_mean']
        visual=(aligned,fg['rgb']) if return_aligned else fg['rgb']
        geometry=mask_geometry(fg)
        return action,{'inference_ms':(time.perf_counter()-tick)*1000,'mask_pixels':counts,
                       'mask_centroids':{k:v['centroid'] for k,v in geometry.items()},
                       'mask_bboxes':{k:v['bbox'] for k,v in geometry.items()},
                       'foreground_confidence':fg['mean_confidence']},visual


def vision_monitor(frames,output_directory,show=True,fps=20):
    """Record/display policy pixels without blocking inference or ServoP."""
    import signal
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    output=Path(output_directory);output.mkdir(parents=True,exist_ok=False)
    writer=None;metadata=(output/'frames.jsonl').open('x',encoding='utf-8')
    first_stamp=None;last_frame=None;written=0;window='ACT vision: aligned Top | ACT input'
    try:
        while True:
            item=frames.get()
            if item is None:break
            frame=item['frame'];stamp=int(item['stamp_ns'])
            if writer is None:
                height,width=frame.shape[:2]
                writer=cv2.VideoWriter(str(output/'act_vision.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
                if not writer.isOpened():raise OSError('Could not open ACT vision video writer')
                first_stamp=stamp
            target=max(written,round((stamp-first_stamp)*fps/1e9))
            while last_frame is not None and written<target:
                writer.write(last_frame);written+=1
            writer.write(frame);written+=1;last_frame=frame
            metadata.write(json.dumps({k:v for k,v in item.items() if k!='frame'},separators=(',',':'))+'\n')
            metadata.flush()
            if show:
                cv2.imshow(window,frame)
                if cv2.waitKey(1)&0xff==ord('q'):
                    cv2.destroyWindow(window);show=False
    except BaseException as exc:
        (output/'monitor_error.txt').write_text(f'{type(exc).__name__}: {exc}\n')
    finally:
        if writer is not None:writer.release()
        metadata.close()
        if show:cv2.destroyAllWindows()


def policy_worker(requests,responses,config,root,preview_directory=None,vision_frames=None):
    """GPU work has its own process; servo timing never waits for inference."""
    import base64
    import signal
    import traceback
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    try:
        policy=LivePolicy(config,root)
        preview_path=Path(preview_directory) if preview_directory else None
        if preview_path:preview_path.mkdir(parents=True,exist_ok=False)
        last_preview_ns=0
        responses.put({'event':'ready','step':policy.step})
        while True:
            request=requests.get()
            if request is None:return
            try:
                jpeg=np.frombuffer(base64.b64decode(request['jpeg']),np.uint8)
                bgr=cv2.imdecode(jpeg,cv2.IMREAD_COLOR)
                if bgr is None:raise ForegroundUnavailable('Invalid Top compressed image')
                action,metrics,visuals=policy.predict(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),request['state'],request['info'],return_aligned=True)
            except ForegroundUnavailable as exc:
                # Match LimX/RoboTwin rollout semantics: one unusable observation
                # does not terminate the episode.  The control process holds the
                # last action and keeps requesting later observations.
                responses.put({'event':'observation_rejected','stamp_ns':request['stamp_ns'],
                               'sequence':request['sequence'],'reason':str(exc)})
                continue
            aligned,foreground=visuals
            responses.put({'event':'prediction','stamp_ns':request['stamp_ns'],'sequence':request['sequence'],
                'p':request['p'],'r':request['r'],'state':request['state'],'action':action,'completed_ns':time.monotonic_ns(),**metrics})
            if vision_frames is not None:
                panel=np.concatenate([cv2.cvtColor(aligned,cv2.COLOR_RGB2BGR),cv2.cvtColor(foreground,cv2.COLOR_RGB2BGR)],axis=1)
                try:vision_frames.put_nowait({'stamp_ns':request['stamp_ns'],'sequence':request['sequence'],
                    'inference_ms':metrics['inference_ms'],'mask_pixels':metrics['mask_pixels'],
                    'mask_centroids':metrics['mask_centroids'],'mask_bboxes':metrics['mask_bboxes'],'frame':panel})
                except queue.Full:pass
            # Disk I/O stays in the GPU worker, outside the ServoP sender.
            if preview_path and request['stamp_ns']-last_preview_ns>=1_000_000_000:
                pair=np.concatenate([cv2.cvtColor(aligned,cv2.COLOR_RGB2BGR),cv2.cvtColor(foreground,cv2.COLOR_RGB2BGR)],axis=1)
                if not cv2.imwrite(str(preview_path/f"{request['stamp_ns']}.jpg"),pair):raise OSError('Could not save policy preview')
                last_preview_ns=request['stamp_ns']
    except BaseException as exc:
        responses.put({'event':'error','error':str(exc),'traceback':traceback.format_exc()})
