variable "instance_type" {
  description = "EC2 instance type. g4dn.xlarge: 1x T4 GPU, 4 vCPU, 16 GB RAM, ~$0.53/hr. Good for full pipeline."
  default     = "g4dn.xlarge"
}

variable "key_name" {
  description = "Name of an existing EC2 key pair in us-west-2 for SSH access."
  default     = "mac"
}

variable "node_count" {
  description = "Number of identical instances to create (all the same instance_type). 1 for the default single-node workflow; >1 for multi-node DDP -- e.g. 2x g4dn.xlarge (1 GPU, 4 vCPU each) to fit under a 'Running On-Demand G and VT instances' quota too low for a single multi-GPU instance (those jump straight to 48 vCPUs)."
  type        = number
  default     = 1
}
