#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IP=$(terraform -chdir="$REPO_ROOT/terraform" output -raw public_ip 2>/dev/null)
if [[ -z "$IP" ]]; then
  IP=$(terraform -chdir="$REPO_ROOT/terraform" state show aws_instance.census 2>/dev/null \
       | awk '/^\s+public_ip\s*=/ {print $3}' | tr -d '"')
fi
if [[ -z "$IP" ]]; then
  echo "Error: could not determine public IP from terraform" >&2
  exit 1
fi
SSH_OPTS="ssh -i ~/.ssh/mac.pem -o StrictHostKeyChecking=no"

echo "Pushing code to $IP..."
rsync -av \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='terraform' \
  -e "$SSH_OPTS" \
  "$REPO_ROOT/" \
  "ubuntu@$IP:~/ai-for-biology/"

echo "Pulling results from $IP..."
$SSH_OPTS "ubuntu@$IP" "mkdir -p ~/ai-for-biology/results"
rsync -av \
  -e "$SSH_OPTS" \
  "ubuntu@$IP:~/ai-for-biology/results/" \
  "$REPO_ROOT/results/"
