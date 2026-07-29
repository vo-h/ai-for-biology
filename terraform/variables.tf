variable "instance_type" {
  description = "EC2 instance type. g4dn.xlarge: 1x T4 GPU, 4 vCPU, 16 GB RAM, ~$0.53/hr. Good for full pipeline."
  default     = "g4dn.xlarge"
}

variable "key_name" {
  description = "Name of an existing EC2 key pair in us-west-2 for SSH access."
  default     = "mac"
}
