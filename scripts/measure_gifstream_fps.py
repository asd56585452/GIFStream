"""Re-measure GIFStream cam00 render FPS from the delivered bitstream, with warmup (no retraining).

Model = decompress(<gop>/compression/rank0) exactly as run_compression() evaluates it, but the
existing compression/ directory is only read. All val frames (cam00) are moved to the GPU before
timing; timed region = rasterize_splats (anchor decode MLPs + render), torch.cuda.synchronize()
around each call. Results go to --out (a new directory), never to the original GOP directory.

Usage (inside asdggg/gifstream, repo mounted at /workspace):
  python3 scripts/measure_gifstream_fps.py --gop_dir .../flame_steak/GOP_0/r0 --out .../fps/flame_steak_GOP_0
"""
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, '/workspace/examples'); sys.path.insert(0, '/workspace')
import torch, yaml
import simple_trainer_GIFStream as T

p = argparse.ArgumentParser()
p.add_argument('--gop_dir', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--warmup', type=int, default=20)
p.add_argument('--repeats', type=int, default=5)
a = p.parse_args()
if a.out.resolve() == a.gop_dir.resolve() or a.gop_dir.resolve() in a.out.resolve().parents:
    sys.exit('--out must not be inside the original GOP directory')

saved = yaml.unsafe_load(open(a.gop_dir / 'cfg.yml'))
cfg = T.Config()
for k, v in saved.items():
    setattr(cfg, k, v)
cfg.result_dir = str(a.out); cfg.disable_viewer = True; cfg.ckpt = None; cfg.compression = "end2end"  # decoder object only; compress() is never called
runner = T.Runner(0, 0, 1, cfg)

ckpt = torch.load(a.gop_dir / 'ckpts/ckpt_29999_rank0.pt', map_location=runner.device, weights_only=False)
for k in runner.splats.keys():
    runner.splats[k].data = ckpt['splats'][k]
runner.decoders.load_state_dict(ckpt['decoders'])
runner.cfg.compression_sim = ckpt['compression_sim']
assert runner.cfg.compression_sim
runner.load_entropy_model_from_ckpt(ckpt, cfg.entropy_model_type)
compress_dir = str(a.gop_dir / 'compression/rank0')
# Same steps as the evaluation half of run_compression(), read-only on compress_dir.
runner.load_models_from_compressed_dir(compress_dir, cfg.entropy_model_type)
splats_c = runner.compression_method.decompress(compress_dir, runner.entropy_models, runner.device)
for k in splats_c.keys():
    runner.splats[k].data = splats_c[k].to(runner.device)
if cfg.knn:
    _, runner.indices = T.find_k_neighbors(runner.splats['anchors'], cfg.n_knn)
runner.comp_sim_splats, runner.esti_bits_dict = runner.compression_sim_method.simulate_compression(
    runner.splats, cfg.max_steps, 0, cfg.entropy_channel)
runner.istraining = False

frames = []
for d in torch.utils.data.DataLoader(runner.valset, batch_size=1, shuffle=False, num_workers=0):
    frames.append(dict(c2w=d['camtoworld'].to(runner.device), K=d['K'].to(runner.device),
                       cam=d['camera_id'].to(runner.device), t=float(d['time']),
                       gt=d['image'].to(runner.device) / 255.0))
H, W = frames[0]['gt'].shape[1:3]

@torch.no_grad()
def render(f):
    out, _, _ = runner.rasterize_splats(camtoworlds=f['c2w'], Ks=f['K'], width=W, height=H, sh_degree=None,
                                        near_plane=cfg.near_plane, far_plane=cfg.far_plane, masks=None,
                                        time=f['t'], camera_ids=f['cam'])
    return out

with torch.no_grad():
    psnrs = []
    for f in frames:
        img = render(f).clamp(0, 1)
        psnrs.append((-10 * torch.log10(((img - f['gt']) ** 2).mean())).item())
    for i in range(a.warmup):
        render(frames[i % len(frames)])
    samples = []
    for _ in range(a.repeats):
        for f in frames:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            render(f)
            torch.cuda.synchronize(); samples.append(time.perf_counter() - t0)

ref = json.load(open(a.gop_dir / 'stats/compress_step29999.json'))
mean_psnr = sum(psnrs) / len(psnrs)
rec = dict(gop_dir=str(a.gop_dir), width=int(W), height=int(H), num_frames=len(frames),
           warmup_renders=a.warmup, repeats=a.repeats, num_samples=len(samples),
           mean_seconds=sum(samples) / len(samples), render_fps=len(samples) / sum(samples),
           old_ellipse_time=ref.get('ellipse_time'), old_fps=1 / ref['ellipse_time'] if ref.get('ellipse_time') else None,
           check_psnr=mean_psnr, ref_compress_psnr=ref['psnr'],
           definition='decompressed bitstream; cam00 val frames preloaded on GPU; rasterize_splats incl. anchor MLP decode; synchronized; warmup excluded')
if not math.isfinite(mean_psnr) or abs(mean_psnr - ref['psnr']) > 0.05:
    rec['error'] = 'decompressed render PSNR does not match recorded compress PSNR'
a.out.mkdir(parents=True, exist_ok=True)
(a.out / 'render_fps.json').write_text(json.dumps(rec, indent=2))
print('RESULT ' + json.dumps({k: rec[k] for k in ('render_fps', 'old_fps', 'check_psnr', 'ref_compress_psnr', 'width', 'height', 'num_frames')}))
if 'error' in rec:
    sys.exit(rec['error'])
