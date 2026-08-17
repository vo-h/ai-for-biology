output "public_ip" {
  # Node 0 -- keeps single-node tooling (scripts/sync.sh, most of
  # scripts/run_remote.sh) working unchanged regardless of node_count.
  value = aws_instance.census[0].public_ip
}

output "public_ips" {
  value = aws_instance.census[*].public_ip
}

output "private_ips" {
  # Multi-node torchrun's --rdzv-endpoint needs the rendezvous host's
  # *private* IP -- instance-to-instance traffic inside the VPC, not routed
  # out through the public internet.
  value = aws_instance.census[*].private_ip
}

output "ssh" {
  value = "ssh -i ~/.ssh/mac.pem ubuntu@${aws_instance.census[0].public_ip}"
}

output "rsync" {
  value = "rsync -av --exclude='.venv' --exclude='results' --exclude='__pycache__' -e 'ssh -i ~/.ssh/mac.pem' ./ ubuntu@${aws_instance.census[0].public_ip}:~/ai-for-biology/"
}
