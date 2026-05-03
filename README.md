<h1 align="center">
  <br>
  <img src="https://drive.google.com/uc?id=1Mz2WqXHrwEOjwtWfJVHV7NiRwC_64Shh">
</h1>
<p align="center">
  <a href="#modules">Modules</a> •
  <a href="#code-structure">Code structure</a> •
  <a href="#code-design">Code design</a> •
  <a href="#installing-the-application">Installing the application</a> •
  <a href="#taskfile-commands">Taskfile commands</a> •
  <a href="#environments">Environments</a> •
  <a href="#running-the-application">Running the application</a> •
  <a href="#changelog">Changelog</a>
</p>


--------------------------------------------------------------------------------

Sakura is a simple but powerful tool to reduce training time by running the train/test asynchronously. It provides two features:
- A simple ML framework for asynchronous training.
- An integration with PyTorch Lightning.

You can reuse your favorite Python framework such as PyTorch, TensorFlow or PaddlePaddle.


# Modules

At a granular level, Sakura is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **sakura/** | Core package — version metadata, config, and functional utilities |
| **sakura/ml/** | Machine-learning primitives: async trainer, Sakura trainer, epoch range |
| **sakura/lightning/** | PyTorch Lightning integration (`SakuraTrainer`) |
| **docker/** | Multi-stage Dockerfile and s6-overlay service scripts |
| **taskfiles/** | Modular Taskfile includes for build, test, bench, and docker workflows |
| **tests/** | Pytest test suite for trainers, epoch range, functional utils, and packaging |
| **mnist_demo/** | Standalone MNIST demo with Lightning trainer |


# Code structure

```
.
├── main.py                          # Benchmark CLI: baseline vs Sakura trainer
├── pyproject.toml                   # PEP 621 project metadata (hatchling)
├── Taskfile.yml                     # Task runner entry point
├── compose.yaml                     # Docker Compose (production + dev services)
├── .env.template                    # Environment variable template
├── docker/
│   ├── Dockerfile                   # Multi-stage: base → test → production
│   └── static/
│       ├── redis/run                # s6 service script for Redis
│       └── sakura/run               # s6 service script for Sakura
├── sakura/
│   ├── __init__.py                  # Package version and build metadata
│   ├── __main__.py                  # CLI entry point (python -m sakura)
│   ├── config.yaml                  # Default configuration
│   ├── functional.py                # Metric namespaces and defaults
│   ├── ml/
│   │   ├── async_trainer.py         # Asynchronous training loop
│   │   ├── sakura_trainer.py        # Core Sakura trainer
│   │   └── epoch/
│   │       └── range.py             # Epoch range utilities
│   └── lightning/
│       └── __init__.py              # SakuraTrainer (Lightning integration)
├── taskfiles/
│   ├── bench.yml                    # Benchmark tasks
│   ├── build.yml                    # Wheel/sdist build tasks
│   ├── docker.yml                   # Docker build/run/shell tasks
│   └── test.yml                     # Pytest tasks (local + Docker)
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures
│   ├── test_async_trainer.py
│   ├── test_epoch_range.py
│   ├── test_functional.py
│   ├── test_pkg_info.py
│   └── test_sakura_trainer.py
└── mnist_demo/
    └── lightning/
        └── main.py                  # Standalone MNIST Lightning example
```

# Code design

Sakura wraps your existing PyTorch Lightning workflow. Simply swap `L.Trainer` for `SakuraTrainer` and call `trainer.run(...)` instead of `trainer.fit(...)`. The framework handles asynchronous test evaluation behind the scenes via Redis so that validation does not block training.

```python
from sakura.lightning import SakuraTrainer

model = MNISTModel()
trainer = SakuraTrainer(accelerator="auto", max_epochs=10)
trainer.run(model, train_loader, val_loader, model_path="models/best_model.pth")
```

# Installing the application

To clone and run this application, you'll need the following installed on your computer:
- [Git](https://git-scm.com)
- [Python >= 3.10](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- [Task](https://taskfile.dev/) (optional, for task runner commands)
- Docker Desktop (optional, for containerised builds)
   - [Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/)
   - [Install Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Install Docker Desktop on Linux](https://docs.docker.com/desktop/install/linux-install/)

### Clone and install

```bash
# Clone this repository
git clone https://github.com/zakuro-ai/sakura
cd sakura

# Copy environment template
cp .env.template .env

# Install in editable mode
uv pip install -e .
```

### Verify installation

```bash
python3 -c "import sakura; print(sakura.__version__)"
```

# Taskfile commands

This project uses [Task](https://taskfile.dev/) as its task runner. Available commands:

| Command | Description |
| --- | --- |
| `task` | Show available tasks |
| `task help` | Display project version and usage info |
| `task setup` | Install prerequisites and verify environment |
| `task build:wheel` | Build wheel distribution with uv |
| `task build:sdist` | Build source distribution with uv |
| `task build:all` | Build wheel and source distribution |
| `task build:clean` | Remove dist directory |
| `task test:run` | Run pytest test suite |
| `task test:docker` | Run pytest test suite inside a Docker container |
| `task docker:build-image` | Build the Docker image via docker compose |
| `task docker:build` | Full build (wheel + Docker image) |
| `task docker:run` | Run services via docker compose |
| `task docker:shell` | Launch docker compose and open a bash shell |
| `task bench:baseline` | Benchmark baseline Lightning trainer |
| `task bench:sakura` | Benchmark Sakura trainer |
| `task bench:both` | Benchmark both trainers and compare |
| `task bench:docker` | Run benchmark inside the sakura-dev container |

# Environments

Environment variables are defined in `.env` (copied from `.env.template`):

| Variable | Description |
| --- | --- |
| `ORG` | Organisation name (default: `zakuroai`) |
| `ZAKURO_LOGS` | Path to log directory |
| `SAKURA_HOME` | Project root directory |

Additional variables used at runtime:

| Variable | Description |
| --- | --- |
| `PATH_DATASETS` | Dataset download directory (default: `/tmp/datasets`) |
| `PYTHONPATH` | Python module search path (set in `compose.yaml`) |
| `MASTER_HOST` | Master node address for distributed training (default: `127.0.0.1`) |

# Running the application

### Benchmark CLI

```bash
# Run both baseline and Sakura trainers (default)
python3 main.py --mode both --epochs 10

# Run only the Sakura trainer
python3 main.py --mode sakura --epochs 5

# Run only the baseline Lightning trainer
python3 main.py --mode baseline
```

### Using the Sakura CLI

```bash
# Launch Sakura via the CLI entry point
sakura main.py
```

You should be able to see this output with no delay between epochs (asynchronous testing).
```
   _____           _                               __  __   _
  / ____|         | |                             |  \/  | | |
 | (___     __ _  | | __  _   _   _ __    __ _    | \  / | | |
  \___ \   / _` | | |/ / | | | | | '__|  / _` |   | |\/| | | |
  ____) | | (_| | |   <  | |_| | | |    | (_| |   | |  | | | |____
 |_____/   \__,_| |_|\_\  \__,_| |_|     \__,_|   |_|  |_| |______|

(0) MNIST | Epoch: 1/10 | Acc: 0.0000 / (0.0000) | Loss:0.0000 / (0.0000): 100%|██████████| 18/18 [00:06<00:00,  2.69it/s]
(1) MNIST | Epoch: 2/10 | Acc: 0.0000 / (0.0000) | Loss:0.0000 / (0.0000): 100%|██████████| 18/18 [00:05<00:00,  3.36it/s]
...
(8) MNIST | Epoch: 9/10 | Acc: 98.2402 / (98.2402) | Loss:0.0886 / (0.0886): 100%|██████████| 18/18 [00:05<00:00,  3.41it/s]
```

FYI the meaning of the above notation is:
```
([best_epoch]) [name_exp] | Epoch: [current]/[total] | Acc: [current_test_acc] / ([best_test_acc]) | Loss:[current_test_loss] / ([best_test_loss]): 100%|███| [batch_k]/[batch_n] [[time_train]<[time_left], [it/s]]
```

### Using Docker

```bash
# Build and run via Task
task docker:build
task docker:run

# Or directly with docker compose
docker compose up sakura -d
docker exec -it sakura bash
```

# Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed list of changes.
