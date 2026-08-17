#!/usr/bin/env bash
set -euo pipefail

# Run a training command on an EC2 instance of a chosen type, streaming its
# stdout/stderr back to a local log file in real time.
#
# The remote command runs detached (setsid+nohup, launched from a small
# runner script written via stdin rather than inlined into a heredoc/command
# string -- avoids nested shell-quoting hazards if your training args
# contain spaces or quotes). It survives this script exiting, your local
# network dropping, or Ctrl-C here, which only stops the local log tail, not
# the remote job. Re-attach later with `--reattach <run-name>` (see below).
#
# `terraform apply` runs with -auto-approve by default (this script is meant
# for repeated iteration, not one-off runs) -- pass --confirm to get the
# interactive "yes" prompt back, e.g. before a change you want to review
# first (a resize/replace of the instance, not just its first creation).
#
# The training command's own outputs (checkpoints, model_config.json,
# run_metadata.json -- wherever its --output-dir points under results/) are
# pulled back automatically whenever the local log stream ends, Ctrl-C or
# not (see the EXIT trap below) -- not just the log. Pull on demand instead
# with --pull-results, e.g. to grab a checkpoint mid-run without touching
# the stream.
#
# --nodes N (default 1) provisions N identical instances instead of one --
# for multi-node DDP, e.g. sidestepping a per-instance vCPU quota too low
# for a single multi-GPU box. Streaming and results-pulling only ever target
# node 0 -- that's where is_main_process()-gated prints and file writes
# happen in this repo's training loops (see src/training/rxrx1.py), so other
# nodes have nothing worth streaming.
#
# For a bare `torchrun ...` command, the rendezvous flags (--standalone for
# one node; --nnodes/--node_rank/--rdzv_id/--rdzv_backend/--rdzv_endpoint
# for multiple) are filled in automatically from --nodes and --name -- they
# don't vary per training script, so there's nothing to type. Put your own
# in instead (--standalone, --rdzv*, --node_rank/--node-rank) and yours win,
# nothing here overrides them. {NODE_RANK} (0-indexed) and {MASTER_ADDR}
# (node 0's private IP) are the placeholders those auto-filled flags expand
# to per node -- usable directly in your own command too, if you need one of
# them somewhere the auto-fill doesn't reach.
#
# Usage:
#   scripts/run_remote.sh --instance-type <type> [--name <run-name>] [--nodes N] [--confirm] -- <command...>
#   scripts/run_remote.sh --reattach <run-name>
#   scripts/run_remote.sh --list
#   scripts/run_remote.sh --pull-results
#
# Examples:
#   scripts/run_remote.sh --instance-type g3.8xlarge --name rxrx1-ddp -- \
#       torchrun --nproc_per_node=2 scripts/rxrx1/train_resnet.py \
#       --cell-types HEPG2 --n-epochs 10
#
#   scripts/run_remote.sh --instance-type g4dn.xlarge -- \
#       python scripts/cellxgene/train_mlp.py --split-strategy donor
#
#   # Multi-node: 2x g4dn.xlarge (1 GPU each) instead of one 48-vCPU
#   # multi-GPU instance that might not fit under a vCPU quota. Rendezvous
#   # flags filled in automatically -- just --nproc_per_node (per-node GPU
#   # count) and the training script/args are yours to specify.
#   scripts/run_remote.sh --instance-type g4dn.xlarge --nodes 2 --name rxrx1-ddp -- \
#       torchrun --nproc_per_node=1 scripts/rxrx1/train_resnet.py --n-epochs 30
#
#   scripts/run_remote.sh --reattach rxrx1-ddp   # resume streaming after a Ctrl-C / dropped connection
#   scripts/run_remote.sh --list                 # forgot the run name? list every run on the instance
#   scripts/run_remote.sh --pull-results         # grab checkpoints/configs without touching the log stream
#
# Teardown is not this script's job -- when a run is done, `cd terraform &&
# terraform destroy` stops billing. Destroying while a job is still running
# kills it along with the instance, same as unplugging it.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/mac.pem}"
# ConnectTimeout bounds the *initial* connection attempt -- without it, a
# stuck TCP handshake (observed intermittently between these EC2 instances)
# hangs the whole script indefinitely with no feedback, rather than failing
# fast enough to retry. ServerAlive* is the separate, complementary case:
# probes every 15s, tolerates up to 4 missed (60s), for a connection that
# was established fine but then goes dead mid-command -- matters most for
# the venv-setup step, which runs synchronously/attached (unlike the actual
# training launch, detached via setsid+nohup) -- a killed `python3 -m venv`
# mid-write leaves a broken venv behind (symlinks written before
# pip/activate), not a clean failure.
SSH_OPTS="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"

# Bounds total wall time for one attempt, not just the connection phase --
# ConnectTimeout only covers the initial TCP handshake, and ServerAlive*
# only catches a connection that's gone fully unresponsive. Neither bounds
# a connection that's alive and answering keepalives but where the actual
# remote command just hangs -- observed happening here. No `timeout` binary
# on macOS by default, so implemented directly: background the command,
# poll for it to finish, kill it if it doesn't within `secs`.
run_with_timeout() {
  local secs="$1"; shift
  "$@" &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [[ "$waited" -ge "$secs" ]]; then
      # $pid is the backgrounded function's own subshell -- killing only
      # that leaves its child (the actual ssh process) orphaned and still
      # running/hung, not actually stopped. Kill the child first.
      pkill -9 -P "$pid" 2>/dev/null
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
    sleep 1
    waited=$((waited + 1))
  done
  wait "$pid"
}

# Retries a flaky SSH-based step (connectivity between these specific EC2
# instances has been observed to intermittently hang, clearing up on a bare
# retry) rather than letting one bad attempt hang the whole script or abort
# it outright. Each attempt hard-capped via run_with_timeout -- per_attempt_secs
# is the first argument, not hardcoded, since steps vary a lot in how long a
# genuine (non-hung) attempt legitimately takes: measured torchrun itself
# taking ~18s just to background (loading torch/CUDA) before a launch is even
# hung/not, well over what a plain sync/venv-check step needs.
ssh_retry() {
  local per_attempt_secs="$1"; shift
  local attempt=1 max=5
  while true; do
    if run_with_timeout "$per_attempt_secs" "$@"; then return 0; fi
    if [[ "$attempt" -ge "$max" ]]; then return 1; fi
    echo "    (attempt $attempt/$max failed/timed out, retrying in 5s...)" >&2
    sleep 5
    attempt=$((attempt + 1))
  done
}

instance_type=""
run_name="run-$(date +%Y%m%d-%H%M%S)"
reattach_name=""
list_mode=0
pull_mode=0
nodes=1
tf_apply_flag="-auto-approve"
remote_cmd=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-type) instance_type="$2"; shift 2 ;;
    --name) run_name="$2"; shift 2 ;;
    --reattach) reattach_name="$2"; shift 2 ;;
    --list) list_mode=1; shift ;;
    --pull-results) pull_mode=1; shift ;;
    --nodes) nodes="$2"; shift 2 ;;
    --confirm) tf_apply_flag=""; shift ;;
    --) shift; remote_cmd=("$@"); break ;;
    -h|--help)
      # Only the leading doc-comment block (after the shebang) -- a plain
      # `grep '^#'` would also pick up comment lines inside the heredocs
      # further down (e.g. the runner script's own "#!/usr/bin/env bash").
      awk 'NR==1{next} /^#/{started=1; sub(/^# ?/,""); print; next} started{exit}' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1 (did you forget '--' before the command?)" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$SSH_KEY" ]]; then
  echo "Error: SSH key not found at $SSH_KEY (override with SSH_KEY=/path/to/key)" >&2
  exit 1
fi

IP=$(terraform -chdir="$REPO_ROOT/terraform" output -raw public_ip 2>/dev/null)

# Pulls the whole remote results/ dir back (not just this run's slice) --
# rsync'd wholesale, same as the original sync.sh, since this script doesn't
# know where a given training command's --output-dir actually points (it's
# just an opaque argument inside remote_cmd). Depends on $IP and
# $REMOTE_HOME already being set by the caller.
pull_results() {
  echo
  echo "==> Pulling results/ from $IP..."
  rsync -av -e "$SSH_OPTS" "ubuntu@$IP:$REMOTE_HOME/ai-for-biology/results/" "$REPO_ROOT/results/" \
    || echo "    (pull failed -- retry later with: scripts/run_remote.sh --pull-results)"
}

# --- Pull-results mode: grab checkpoints/configs on demand, independent of
# the log stream. ---
if [[ "$pull_mode" == "1" ]]; then
  if [[ -z "$IP" ]]; then
    echo "Error: no instance IP from terraform -- is one currently up?" >&2
    exit 1
  fi
  echo "==> Instance IP: $IP"
  REMOTE_HOME=$($SSH_OPTS "ubuntu@$IP" 'echo $HOME')
  pull_results
  exit 0
fi

# --- List mode: forgot the run name -- show every run's name, whether it's
# still running, and when its log was last written to. ---
if [[ "$list_mode" == "1" ]]; then
  if [[ -z "$IP" ]]; then
    echo "Error: no instance IP from terraform -- is one currently up?" >&2
    exit 1
  fi
  echo "==> Instance IP: $IP"
  REMOTE_HOME=$($SSH_OPTS "ubuntu@$IP" 'echo $HOME')
  REMOTE_LOG_DIR="$REMOTE_HOME/ai-for-biology/results/logs"
  echo "==> Runs on $IP:"
  $SSH_OPTS "ubuntu@$IP" bash -s <<EOF
cd "$REMOTE_LOG_DIR" 2>/dev/null || { echo "(no runs yet -- $REMOTE_LOG_DIR doesn't exist)"; exit 0; }
shopt -s nullglob
logs=(*.log)
if [ \${#logs[@]} -eq 0 ]; then
  echo "(no runs found)"
  exit 0
fi
printf '%-28s %-9s %s\n' NAME STATUS "LAST MODIFIED"
for f in "\${logs[@]}"; do
  name="\${f%.log}"
  pidfile="\${name}.pid"
  status=stopped
  if [ -f "\$pidfile" ] && kill -0 "\$(cat "\$pidfile")" 2>/dev/null; then
    status=running
  fi
  mtime=\$(date -r "\$f" "+%Y-%m-%d %H:%M:%S")
  printf '%-28s %-9s %s\n' "\$name" "\$status" "\$mtime"
done
EOF
  echo
  echo "Reattach with: scripts/run_remote.sh --reattach <name>"
  exit 0
fi

# --- Reattach mode: skip provision/sync/launch, just resume streaming an
# already-running job's log. ---
if [[ -n "$reattach_name" ]]; then
  if [[ -z "$IP" ]]; then
    echo "Error: no instance IP from terraform -- is one currently up? (terraform -chdir=terraform output public_ip)" >&2
    exit 1
  fi
  echo "==> Instance IP: $IP"
  REMOTE_HOME=$($SSH_OPTS "ubuntu@$IP" 'echo $HOME')
  REMOTE_LOG="$REMOTE_HOME/ai-for-biology/results/logs/${reattach_name}.log"
  LOCAL_LOG="$REPO_ROOT/results/logs/${reattach_name}.log"
  mkdir -p "$REPO_ROOT/results/logs"

  echo "==> Reattaching: $REMOTE_LOG -> $LOCAL_LOG"
  echo "    Ctrl-C stops this local stream only -- the remote job keeps running."
  echo "    Results (checkpoints/configs) get pulled automatically when this ends."
  echo
  # Results pulled on the way out regardless of *how* the stream ends
  # (Ctrl-C, dropped connection, or the pipe just returning) -- an EXIT trap
  # fires on all of those, not just a clean return.
  trap pull_results EXIT
  # -n +1, not just -f, so the local log is the complete log from the start
  # every time, not just whatever's written after this particular attach.
  $SSH_OPTS "ubuntu@$IP" "tail -f -n +1 '$REMOTE_LOG'" | tee "$LOCAL_LOG"
  exit 0
fi

# --- Launch mode ---
if [[ -z "$instance_type" ]]; then
  echo "Error: --instance-type is required (or use --reattach <run-name>)" >&2
  exit 1
fi
if [[ ${#remote_cmd[@]} -eq 0 ]]; then
  echo "Error: no command given after --" >&2
  exit 1
fi

# For a bare `torchrun ...` invocation, fill in the rendezvous boilerplate
# automatically -- it's fully determined by --nodes and run_name, nothing
# about it varies per training script. Only fills gaps: if you've already
# put --standalone/--rdzv*/--node_rank/--node-rank in yourself, nothing here
# touches your command. {NODE_RANK}/{MASTER_ADDR} still work as manual
# placeholders anywhere else in your command (not just inside these
# injected flags) if you need the substitution for something of your own.
if [[ "${remote_cmd[0]}" == "torchrun" ]]; then
  rdzv_configured=0
  for tok in "${remote_cmd[@]}"; do
    case "$tok" in
      --standalone|--rdzv*|--node_rank*|--node-rank*) rdzv_configured=1 ;;
    esac
  done
  if [[ "$rdzv_configured" == "0" ]]; then
    if [[ "$nodes" -eq 1 ]]; then
      remote_cmd=("torchrun" "--standalone" "${remote_cmd[@]:1}")
    else
      # Two separate, independently-configurable timeouts govern different
      # phases -- confirmed by reading torch's own source
      # (c10d_rendezvous_backend.py), not assumed:
      #   read_timeout (default 60s) -- how long a worker's *initial* TCPStore
      #     connection attempt gets before _create_tcp_store gives up outright
      #     and raises RendezvousConnectionError. This is what actually killed
      #     a real run: it crashed in ~2 minutes, nowhere near either 600s or
      #     even the old join_timeout bump -- read_timeout was still at its
      #     60s default.
      #   join_timeout (default 600s) -- separately, how long the rendezvous
      #     state machine waits for enough participants *after* the store
      #     connection is already established.
      # Both bumped for real headroom, not tuned reactively after each miss.
      remote_cmd=(
        "torchrun" "--nnodes=$nodes" "--node_rank={NODE_RANK}"
        "--rdzv_id=$run_name" "--rdzv_backend=c10d" "--rdzv_endpoint={MASTER_ADDR}:29500"
        "--rdzv_conf=join_timeout=1800,read_timeout=1800"
        "${remote_cmd[@]:1}"
      )
    fi
    echo "==> Auto-filled torchrun rendezvous flags: ${remote_cmd[*]}"
  fi
fi

echo "==> Ensuring instance(s) are up (instance_type=$instance_type, nodes=$nodes)..."
terraform -chdir="$REPO_ROOT/terraform" apply -var="instance_type=$instance_type" -var="node_count=$nodes" $tf_apply_flag

IP=$(terraform -chdir="$REPO_ROOT/terraform" output -raw public_ip 2>/dev/null)
if [[ -z "$IP" ]]; then
  echo "Error: could not determine public IP from terraform" >&2
  exit 1
fi
echo "==> Node 0 IP: $IP"

# Full per-node IP lists -- fetched unconditionally (not just when nodes>1)
# so the sync/venv/launch steps below are one code path, a loop of 1 in the
# common single-node case, rather than a branch duplicating that logic.
PUBLIC_IPS=()
while IFS= read -r line; do PUBLIC_IPS+=("$line"); done < <(
  terraform -chdir="$REPO_ROOT/terraform" output -json public_ips | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)))'
)
PRIVATE_IPS=()
while IFS= read -r line; do PRIVATE_IPS+=("$line"); done < <(
  terraform -chdir="$REPO_ROOT/terraform" output -json private_ips | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)))'
)
if [[ "$nodes" -gt 1 ]]; then
  echo "==> All nodes: ${PUBLIC_IPS[*]}"
  echo "    Rendezvous host (node 0) private IP: ${PRIVATE_IPS[0]}"
  echo "    {NODE_RANK} and {MASTER_ADDR} in your command get substituted per node."
fi

# A freshly-created instance answers SSH before user_data (apt-get install
# python3-venv/etc, then `touch /home/ubuntu/ready` as its last step --
# see terraform/main.tf) has actually finished -- syncing/venv-setup this
# early fails with "ensurepip is not available" because python3-venv isn't
# installed yet, not because anything is actually broken. Poll for that
# marker file instead of assuming SSH-reachable means ready.
echo "==> Waiting for instance setup (user_data) to finish on ${#PUBLIC_IPS[@]} node(s)..."
for node_ip in "${PUBLIC_IPS[@]}"; do
  waited=0
  until $SSH_OPTS -o ConnectTimeout=5 "ubuntu@$node_ip" '[ -f /home/ubuntu/ready ]' 2>/dev/null; do
    if [[ "$waited" -ge 300 ]]; then
      echo "Error: $node_ip not ready after 300s -- user_data may have failed. Check with:" >&2
      echo "  ssh -i $SSH_KEY ubuntu@$node_ip 'cat /var/log/cloud-init-output.log'" >&2
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
done
echo "    All node(s) ready."

# Resolve the remote home dir explicitly rather than relying on "~" in
# variables built here: "~" only tilde-expands when it's literal source text
# freshly parsed by a shell, not when substituted from an already-expanded
# local variable -- building absolute paths up front sidesteps that class of
# bug entirely, for every command below.
REMOTE_HOME=$($SSH_OPTS "ubuntu@$IP" 'echo $HOME')
REMOTE_REPO="$REMOTE_HOME/ai-for-biology"
REMOTE_LOG_DIR="$REMOTE_REPO/results/logs"
REMOTE_LOG="$REMOTE_LOG_DIR/${run_name}.log"
REMOTE_PIDFILE="$REMOTE_LOG_DIR/${run_name}.pid"
REMOTE_RUNNER="$REMOTE_LOG_DIR/${run_name}.sh"
LOCAL_LOG_DIR="$REPO_ROOT/results/logs"
LOCAL_LOG="$LOCAL_LOG_DIR/${run_name}.log"
mkdir -p "$LOCAL_LOG_DIR"

# Each step below is a function, not an inline loop body -- ssh_retry needs
# a plain command to re-invoke, and a heredoc-bearing SSH call can't be
# passed through "$@" directly.
sync_node() {
  local node_ip="$1"
  rsync -av \
    --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.git' --exclude='terraform' --exclude='results' \
    --exclude='docs' --exclude='notebooks' --exclude='projects' \
    -e "$SSH_OPTS" \
    "$REPO_ROOT/" "ubuntu@$node_ip:$REMOTE_REPO/"
}

setup_venv_on_node() {
  local node_ip="$1"
  $SSH_OPTS "ubuntu@$node_ip" bash -s <<EOF
set -e
mkdir -p "$REMOTE_LOG_DIR"
cd "$REMOTE_REPO"
# Check for .venv/bin/pip, not just the .venv directory existing -- venv
# creation makes the directory before running ensurepip, so a run that hit
# the user_data race (python3-venv not installed yet) leaves a real but
# broken .venv behind. A bare directory-existence check would skip past it
# and fail differently and more confusingly, in pip install below.
[ -f .venv/bin/pip ] || { rm -rf .venv; python3 -m venv .venv; }
source .venv/bin/activate
pip install -q -r requirements.txt
EOF
}

write_runner_on_node() {
  local node_ip="$1" node_cmd_str="$2"
  $SSH_OPTS "ubuntu@$node_ip" "cat > '$REMOTE_RUNNER'" <<SCRIPT
#!/usr/bin/env bash
set -e
cd "$REMOTE_REPO"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
exec $node_cmd_str
SCRIPT
}

launch_node() {
  local node_ip="$1" i="$2"
  # Checks for an already-alive process from a *previous* attempt before
  # launching another. Belt-and-suspenders with -n/the longer timeout below:
  # those reduce how often ssh_retry wrongly treats a successful launch as
  # hung, but starting a background job isn't naturally idempotent the way
  # syncing files or writing a config is, and a retry -- triggered for
  # *any* reason, not just the specific one already diagnosed -- must not
  # be able to start a second training process alongside the first.
  # Confirmed happening for real: found 4 duplicate torchrun processes on
  # one node from exactly this before this check existed.
  #
  # -n (redirect ssh's own stdin from /dev/null): the backgrounded remote
  # command already redirects its own stdin/stdout/stderr away from the ssh
  # channel, but ssh's *client-side* stdin was still connected -- classic
  # "ssh hangs after backgrounding a process" gotcha, distinct from the
  # remote command's own redirects. Only applied here, not in $SSH_OPTS
  # globally -- write_runner_on_node/setup_venv_on_node need real stdin for
  # their heredocs.
  $SSH_OPTS -n "ubuntu@$node_ip" \
    "if [ -f '$REMOTE_PIDFILE' ] && kill -0 \"\$(cat '$REMOTE_PIDFILE')\" 2>/dev/null; then
       echo \"node $i already running (pid \$(cat '$REMOTE_PIDFILE')), not relaunching\"
     else
       chmod +x '$REMOTE_RUNNER'
       setsid nohup '$REMOTE_RUNNER' > '$REMOTE_LOG' 2>&1 < /dev/null &
       echo \$! > '$REMOTE_PIDFILE'
       sleep 1
       echo \"node $i launched\"
     fi"
}

fail_node() {
  echo "Error: $1 unreachable after retries on $2 -- check the instance/your network and try again." >&2
  exit 1
}

echo "==> Syncing code to ${#PUBLIC_IPS[@]} node(s)..."
for node_ip in "${PUBLIC_IPS[@]}"; do
  ssh_retry 60 sync_node "$node_ip" || fail_node "$node_ip" "code sync"
done

echo "==> Ensuring venv + dependencies on ${#PUBLIC_IPS[@]} node(s)..."
for node_ip in "${PUBLIC_IPS[@]}"; do
  # 600s, not the default -- a first-time `pip install` of this
  # requirements.txt (torch, cellxgene-census, tiledbsoma, scanpy, ...) on a
  # fresh venv measured several minutes on its own. A short timeout here
  # would treat a legitimately-still-installing venv as hung and retry into
  # it, on top of everything already established about retries into
  # in-progress work not being free here.
  ssh_retry 600 setup_venv_on_node "$node_ip" || fail_node "$node_ip" "venv setup"
done

echo "==> Launching (detached) on ${#PUBLIC_IPS[@]} node(s): ${remote_cmd[*]}"
# Workers (rank 1+) launch before the master (rank 0), not in index order --
# the master's TCPStore server starts its own "waiting for clients" clock
# the moment it opens its listening socket, and every worker's *initial*
# connection attempt is bounded by the separate, short-by-default
# read_timeout (60s, see the --rdzv_conf note above) before it gives up
# outright. Getting workers into their retry loop first means the master's
# socket opens into workers that are already retrying, instead of starting
# both clocks near-simultaneously and hoping they overlap.
launch_order=()
for ((idx = ${#PUBLIC_IPS[@]} - 1; idx >= 0; idx--)); do launch_order+=("$idx"); done
for i in "${launch_order[@]}"; do
  node_ip="${PUBLIC_IPS[$i]}"

  # {NODE_RANK}/{MASTER_ADDR} substituted per node, on the raw tokens
  # captured after "--", before %q-escaping -- e.g. a multi-node torchrun's
  # --node_rank and --rdzv_endpoint. Every other node gets the identical
  # command otherwise.
  node_cmd=()
  for tok in "${remote_cmd[@]}"; do
    tok="${tok//\{NODE_RANK\}/$i}"
    tok="${tok//\{MASTER_ADDR\}/${PRIVATE_IPS[0]}}"
    node_cmd+=("$tok")
  done
  # %q-quotes each arg into a string bash can safely re-parse back into the
  # same argv later. Written to a remote runner script via stdin (`cat >
  # file`, pure byte copy, no shell re-parsing of the content by either
  # side) rather than inlined into a heredoc or command string -- avoids
  # nested quoting hazards if a training-script argument contains spaces or
  # quotes.
  printf -v node_cmd_str '%q ' "${node_cmd[@]}"
  ssh_retry 30 write_runner_on_node "$node_ip" "$node_cmd_str" || fail_node "$node_ip" "writing the runner script"
  # 90s, not 30 -- measured torchrun's own startup (loading torch/CUDA)
  # taking ~18s just to background, before any network variance on top of
  # that. 30s left too little margin and produced a real, observed bug:
  # a launch that actually succeeded got treated as hung, retried, and
  # launched a *second* training process alongside the first (starting a
  # background job isn't naturally idempotent the way syncing files is).
  ssh_retry 90 launch_node "$node_ip" "$i" || fail_node "$node_ip" "launching"
done

echo "==> Streaming node 0's $REMOTE_LOG -> $LOCAL_LOG"
echo "    Ctrl-C stops this local stream only -- the remote job keeps running."
echo "    Reattach any time with:"
echo "      scripts/run_remote.sh --reattach $run_name"
echo "    Stop the remote job with (best-effort -- pidfile may point at a"
echo "    wrapper process; pkill -f the training script name if this doesn't work):"
echo "      ssh -i $SSH_KEY ubuntu@$IP 'kill \$(cat $REMOTE_PIDFILE)'"
echo "    Tear down the instance(s) when you're done (kills any still-running job):"
echo "      cd terraform && terraform destroy"
echo "    Results (checkpoints/configs) get pulled automatically when this stream ends."
if [[ "$nodes" -gt 1 ]]; then
  echo "    Only node 0 is streamed/pulled -- that's where this repo's training loops"
  echo "    gate prints/file writes (is_main_process()). Other nodes' logs, if you need"
  echo "    them (e.g. to debug a rendezvous hang), are at the same path on their own IPs:"
  echo "      ${PUBLIC_IPS[*]:1}"
fi
echo

# Results pulled on the way out regardless of *how* the stream ends (Ctrl-C,
# dropped connection, or the pipe just returning) -- an EXIT trap fires on
# all of those, not just a clean return.
trap pull_results EXIT
# -n +1, not just -f, so the local log captures everything written from the
# very start of the run, not just what's written after tail happens to attach.
$SSH_OPTS "ubuntu@$IP" "tail -f -n +1 '$REMOTE_LOG'" | tee "$LOCAL_LOG"
