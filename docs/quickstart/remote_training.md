# Remote training over SSH

AlphaBrain runs training on the same machine as the UI by default. An
administrator can instead send every training job to one SSH server while the
UI continues to own the queue, job status, and logs.

## Prerequisites

- The remote server has a working AlphaBrain checkout and its training
  dependencies.
- The machine running the UI can connect non-interactively with an SSH key or
  `ssh-agent`.
- The server host key is already present in the UI machine's `known_hosts`
  file. AlphaBrain does not disable host-key verification.
- Datasets, pretrained models, and checkpoints referenced by a recipe are
  available at the paths used by the remote checkout.

Test the connection once from the machine running the UI:

```bash
ssh user@gpu-server.example.edu 'cd /srv/AlphaBrain && nvidia-smi'
```

## Configure the UI

Open **Settings → Environment → Remote training** and set:

- **SSH host, user, and port** for the training server.
- **Remote AlphaBrain repository** to its absolute checkout path, such as
  `/srv/AlphaBrain`.
- **Remote GPU IDs** to the devices this AlphaBrain instance may schedule.
- **SSH identity file** only when the default SSH key or `ssh-agent` should not
  be used. This is a path on the UI machine; private-key contents are never
  stored in AlphaBrain settings.
- **Remote environment setup** when the checkout needs activation, for example
  `source .venv/bin/activate` or `source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate alphabrain`.

Enable **Run training over SSH** and save. New training jobs use the remote
server. Deployments, evaluations, downloads, and other utility jobs remain on
the UI machine.

## Runtime behavior

While remote training is enabled, the Overview page adds a **Remote training
server** panel. It polls the configured host over SSH and shows the configured
remote GPUs, CPU and RAM usage, repository-filesystem free space, connection
state, and collection time. The original local GPU, system, and storage panels
remain unchanged. Disabling remote training removes the extra panel.

Generated experiment snapshots are sent through the encrypted SSH session
before the training command starts. They are not passed as command-line
secrets. Standard output and standard error stream back to the normal job log,
and stopping a job terminates the local SSH process and closes its remote
session.

Training outputs stay on the remote server and are not downloaded
automatically. Automatic checkpoint indexing, packaging, deployment, and other
file-based actions require the remote result directory to be mounted on the UI
machine at a usable path. The remote GPU queue coordinates only jobs submitted
through this AlphaBrain UI; it cannot see processes launched independently on
the server.
