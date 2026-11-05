# Independent stages and execution

`run.py` is the root orchestration CLI. It invokes each selected module with
`--plan_json`, reads that public JSON execution plan, and runs the commands in
order. It does not import any stage's algorithm or workflow functions.

Every stage owns its `workflow.py`, arguments, input/output contract and algorithm
code. A stage never imports another stage. `pipeline_runtime.py` contains only
command-plan construction, checked process execution, logging, status, locking,
artifact handling and resume. Small texture-network/query implementations are
kept within their consuming stages: refinement trains its own `network.py`;
registration owns `volume_texture.py`. Compatibility is through the checkpoint
and descriptor schema, rather than a cross-stage Python import.

## Independent CLIs

Run commands from the project root, with the configured Python environment:

```bash
python extraction/run_extraction.py --data_root workspace/person --video_path data/person.mp4 --video_step_size 10 --video_ds_ratio .5
python matting/run_stage.py --data_root workspace/person --device 0
python reconstruction/run_reconstruction.py --data_root workspace/person --device 0 --gs_port 6010
python refinement/run_refinement.py --data_root workspace/person --device 0
python registration/run_registration.py --data_root workspace/person --device 0 --close_eye 0
python texture/run_texture.py --data_root workspace/person/sample_dataset --device 0 --fill_eye_holes --to_metahuman 1
```

`reconstruction/run_colmap.py --data_root workspace/person` runs only camera
calibration. Every module accepts `--resume` and `--plan_json`; the latter only
prints JSON and does not require the input files or GPU models to exist.
The algorithm scripts within a stage retain their individual CLIs as well.

## File contracts

Paths below are relative to the chosen workspace. Supplying these input files
is sufficient to run the corresponding stage independently of earlier code.

| Stage | Inputs | Outputs |
| --- | --- | --- |
| Extraction | video file | `raw_frames/*.png` |
| Matting | raw frames, configured DAViD model | `mask/*.png` |
| Reconstruction | raw frames and masks | `images/`, COLMAP database/model, `recon/`, `2dgs_recon.obj`, `transforms.json` |
| Refinement | raw frames, mesh, camera transforms | selected images, normalized xyz maps, `sample/normalization.json`, `texture/latest.pth`, `appearance.json`, all under `refinement/` |
| Registration | mesh, camera transforms, raw frames, volume checkpoint, selected images; normalization JSON when available | coarse/fine binary PLY geometry and appearance JSON, fitted cameras, frontal RGB/xyz/mask, 3D landmarks, Wrap mesh, `sample_dataset/` |
| Texture | `sample_dataset/final_hack.obj`, images and camera transforms | optional eye-cap mesh, UV buffers, optimized `texture/image/00150/uv.png` (under `eye_filled/` with caps) |
| MetaHuman export | final UV mesh, calibration JSON | `_metahuman.obj` and `.transform.json` |

Registration can consume checkpoints from existing runs: if their normalization
JSON predates this format, it reconstructs the exact original float32 GPU
normalization from the source 2DGS mesh. A supplied normalization JSON must match
the source mesh SHA-256. It writes its own input descriptor under `register/`.

## Continuous texture coordinates

Training uses `u = ((p - offset) * scale + 1) / 2`, with the same normalization
used by the position-map renderer. New runs save those exact float32 reduction
values and the source mesh SHA in `refinement/sample/normalization.json`.

An appearance descriptor contains schema version 1, relative mesh/checkpoint
paths, the checkpoint architecture (`num_levels=16`, `level_dim=4`) and a 4×4
`world_to_texture` matrix. For column vectors, `u = world_to_texture @ [p,1]`.
After a geometry-frame change `p_new = A @ p_old`, the descriptor updates to
`world_to_texture_new = world_to_texture_old @ inverse(A)`. This handles both
coarse scale and FLAME canonical transforms without resampling the color field.

The frontal rasterizer interpolates **xyz**, then queries the nonlinear field
in bounded chunks (65,536 visible pixels each). Background pixels are excluded
using rasterizer face validity, never by testing xyz against zero. This avoids
the detail loss from interpolating baked per-vertex RGB. No vertex densification
or color baking is performed.

Coarse/fine intermediate meshes use binary PLY. Wrap does not accept PLY, so its
interface writes one geometry-only `register/wrap/target_geometry.obj`. This
adapter contains no baked vertex colors. Debug outputs are `frontal_rgb.png`,
`frontal_xyz.pt`, `frontal_mask.png` and `rendered.png` (landmark overlay).

## Logging and failures

The console prints numbered start/finish banners such as
`[21/27] RUN reg.frontal_landmarks`, child progress/tracebacks, and a heartbeat
every 30 seconds for long operations. Every process uses explicit argument
arrays, `cwd` and the active Python executable; no shell command concatenation
or global working-directory mutation is needed.

Each invocation writes:

```text
workspace/person/
  pipeline_status.json            # current/last run, current step, per-step status
  outputs.json                    # available final outputs, or failure state
  logs/latest.json                # path to the latest run directory
  logs/<run_id>/
    pipeline.log                  # combined console stream and stage banners
    01_extract.frames.log         # exact child stdout + stderr, including tqdm
    ...
    events.jsonl                  # structured step/heartbeat/completion events
    status.json                   # run snapshot, argv, cwd, code/input/output records
  history/<run_id>/...            # prior owned artifacts replaced during this run
```

A nonzero return code, missing/nonempty input or missing/stale declared output
fails that named step and stops subsequent steps. The console gives its log
path. Ctrl+C terminates the child process group, records `cancelled`, and releases
the workspace lock. One active invocation per workspace prevents mixed outputs.

Run `python run.py --save_root workspace/person --status` from another terminal,
or follow `logs/<run_id>/pipeline.log` with `tail -f`.

## Resume and artifact generations

Use the same command/options with `--resume`. Reuse requires a previous successful
step with matching argv, cwd, relevant code/dependencies, CUDA device, inputs and
output fingerprints. Existing files alone are insufficient. The first mismatch
reruns that step and all later selected steps. Editing only registration code
therefore preserves earlier valid reconstruction steps. COLMAP's shared database
and in-place model updates are tracked explicitly to avoid false invalidation.

Before a selected step regenerates an owned output tree, the runner moves its
previous version into `history/<new_run_id>/`. This preserves old results while
preventing residual PNGs, sparse models or texture caches from mixing with the
new generation. Source inputs/checkpoints supplied to a later-only invocation
are not removed. To reuse completed work, select only the needed stages via
`--func reg-tex`, or use `--resume` with the original command.
