# PESTO training container

The image uses the same internal Avito Python 3.11 / CUDA 11.7 base and
PyTorch 2.5.1 cu121 as `speech-lab-ml-experiments`. Its dependencies are kept
in `requirements.txt`, separately from the plugin's Python 3.12/cu126 setup.

Put the training WAV files in `data/pesto_train/`. They are ignored by Git and
are mounted in the container as `/data`.

`docker-compose.yaml` mounts these host directories:

- `ml/` → `/workspace/ml/` (read/write; fine-tune, export and eval code);
- `models/` → `/workspace/models/` (read/write; exported ONNX models);
- `data/` → `/workspace/data/` (read-only; evaluation input);
- `data/pesto_train/` → `/data` (read-only);
- `runs/` → `/workspace/runs` (read/write).

Edits to Python code made in the container are immediately written to the
server. Exported models and evaluation results also persist on the host.

Create the host directories once, then build through Compose:

```bash
mkdir -p data/pesto_train runs/finetune_pesto
HOST_UID=$(id -u) HOST_GID=$(id -g) \
  docker compose -f docker/pesto-train/docker-compose.yaml build
```

Start the long-lived container on host GPU `3` (it appears as `cuda:0` inside):

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) GPU_DEVICE=3 \
  docker compose -f docker/pesto-train/docker-compose.yaml up -d
docker exec -it -w /workspace trainloop-pesto-train bash
```

The current fine-tune → offline teacher labels → streaming distillation
workflow, exact WAV split and commands are in the
[fine-tune README](../../ml/pesto/finetune/README.md). In particular, WAV `21`
is entirely held out for distillation validation; offline fine-tune has no
held-out validation inside its training script. Use a new `run-name` for each
experiment. `runs/` persists on the host.

For a long run started from the host, use `docker exec -d` and redirect output
inside the container to `/workspace/runs/<run-name>.log`. This detaches the
training process from SSH; inspect it with `tail -f runs/<run-name>.log` on the
host. `Ctrl+C` then stops only `tail`, not training.

Rebuild the image after changing dependencies in `requirements.txt`.

Compose publishes TensorBoard on server port `18950`. Open
`http://<server>:18950` from your local machine. TensorBoard is already
installed in the image; the TensorFlow package is not needed for these
PyTorch logs.
