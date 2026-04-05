# Wheelfoot Multi-Expert Deploy

## 1. Export models from Isaac Lab

Run your normal play command once:

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Limx-WF-MultiExpert-Flat-v0 \
  --num_envs 128 \
  --load_run 2026-03-22_01-43-49 \
  --checkpoint model_11500.pt
```

This now exports these files under the checkpoint's `exported/` directory:

- `encoder.onnx`
- `policy.onnx`
- `student_policy.onnx`
- `student_policy.pt`
- `deploy_meta.json`

## 2. Prepare deploy folder

Keep these files ready:

- `encoder.onnx`
- `policy.onnx`
- `student_policy.onnx` (optional if you want single-model inference)
- `deploy_meta.json`
- `scripts/deploy/wf_multiexpert_params.yaml`

## 3. Run the controller

```bash
python scripts/deploy/wf_multiexpert_controller.py \
  --model_dir /path/to/exported \
  --config scripts/deploy/wf_multiexpert_params.yaml
```

## 4. Joystick behavior

- `L1 + Y`: start controller
- `L1 + X`: stop controller
- `A` (`trigger_button_index: 0`): trigger one jump
- `B` (`expert_toggle_button_index: 1`): toggle persistent expert mode when needed

By default the controller stays in jump expert mode. During the jump window it sends an active jump command; outside that window it still keeps `env_group=0` with an inactive jump command.
