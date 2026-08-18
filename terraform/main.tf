terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }
}

resource "aws_security_group" "census" {
  name        = "census-sg"
  description = "SSH access for Census compute, plus inter-node traffic for multi-node DDP"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # torchrun's rendezvous (c10d, TCP) and NCCL/gloo's own data-transfer
  # connections both pick ports that aren't fully fixed in advance --
  # self-referencing (only other members of this same security group, never
  # the public internet) rather than trying to enumerate exact ports.
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "census" {
  count                  = var.node_count
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.census.id]

  root_block_device {
    volume_size = 100 # GB — pip cache + model checkpoints + results
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e
    # This AMI runs its own first-boot NVIDIA driver setup via cloud-init,
    # which can hold the dpkg lock for a while after boot -- wait it out
    # rather than let apt-get fail underneath us.
    while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 1; done
    apt-get update -y
    apt-get install -y python3-pip python3-venv git libhdf5-dev pkg-config
    touch /home/ubuntu/ready
  EOF

  tags = {
    Name = "census-compute-${count.index}"
  }
}
