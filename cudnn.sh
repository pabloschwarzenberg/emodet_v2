# Según
# https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.4.1.50_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8=8.4.1.50-1+cuda11.6
sudo apt-get install libcudnn8-dev=8.4.1.50-1+cuda11.6
sudo apt-get install libcudnn8-samples=8.4.1.50-1+cuda11.6