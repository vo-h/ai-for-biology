output "public_ip" {
  value = aws_instance.census.public_ip
}

output "ssh" {
  value = "ssh -i ~/.ssh/mac.pem ubuntu@${aws_instance.census.public_ip}"
}

output "rsync" {
  value = "rsync -av --exclude='.venv' --exclude='results' --exclude='__pycache__' -e 'ssh -i ~/.ssh/mac.pem' ./ ubuntu@${aws_instance.census.public_ip}:~/ai-for-biology/"
}
