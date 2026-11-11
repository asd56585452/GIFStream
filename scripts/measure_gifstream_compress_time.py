"""Time GIFStream bitstream encoding (run_compression's compress + nets.pt) from an existing ckpt.

The original GOP directory is only read. The bitstream is written to --out and its size is
compared with the original compression/rank0 to confirm the same encoding was timed.

Usage (inside asdggg/gifstream, repo mounted at /workspace):
  python3 scripts/measure_gifstream_compress_time.py --gop_dir .../GOP_0/r0 --out .../compress_time/scene_GOP_0
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, '/workspace/examples'); sys.path.insert(0, '/workspace')
import torch, yaml
import simple_trainer_GIFStream as T

p = argparse.ArgumentParser()
p.add_argument('--gop_dir', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.out.resolve() == a.gop_dir.resolve() or a.gop_dir.resolve() in a.out.resolve().parents:
    sys.exit('--out must not be inside the original GOP directory')

saved = yaml.unsafe_load(open(a.gop_dir / 'cfg.yml'))
cfg = T.Config()
for k, v in saved.items():
    setattr(cfg, k, v)
cfg.result_dir = str(a.out); cfg.disable_viewer = True; cfg.ckpt = None; cfg.compression = 'end2end'
runner = T.Runner(0, 0, 1, cfg)
ckpt = torch.load(a.gop_dir / 'ckpts/ckpt_29999_rank0.pt', map_location=runner.device, weights_only=False)
for k in runner.splats.keys():
    runner.splats[k].data = ckpt['splats'][k]
runner.decoders.load_state_dict(ckpt['decoders'])
runner.cfg.compression_sim = ckpt['compression_sim']
assert runner.cfg.compression_sim
runner.load_entropy_model_from_ckpt(ckpt, cfg.entropy_model_type)  # sets comp_sim_splats, scaling

compress_dir = a.out / 'compression/rank0'
compress_dir.mkdir(parents=True, exist_ok=True)
torch.cuda.synchronize(); t0 = time.perf_counter()
runner.compression_method.compress(str(compress_dir), runner.comp_sim_splats, runner.entropy_models,
                                   cfg.entropy_channel, cfg.c_perframe, runner.scaling, cfg.voxel_size)
nets = {'decoders': runner.decoders.state_dict()}
for name, em in runner.entropy_models.items():
    if em is not None:
        nets[name + '_entropy_model'] = em.state_dict()
nets['scaling'] = runner.compression_sim_method.scaling
torch.save(nets, compress_dir / 'nets.pt')
torch.cuda.synchronize(); seconds = time.perf_counter() - t0

size = lambda d: sum(f.stat().st_size for f in Path(d).rglob('*') if f.is_file())
new_b, old_b = size(compress_dir), size(a.gop_dir / 'compression/rank0')
rec = dict(gop_dir=str(a.gop_dir), compress_seconds=seconds, new_bytes=new_b, original_bytes=old_b,
           size_ratio=new_b / old_b,
           definition='compress() + nets.pt save, synchronized wall clock, from ckpt_29999 (same inputs as run_compression)')
(a.out / 'compress_time.json').write_text(json.dumps(rec, indent=2))
print('RESULT ' + json.dumps(rec))
if abs(rec['size_ratio'] - 1) > 0.02:
    sys.exit('bitstream size differs from original by >2%')
