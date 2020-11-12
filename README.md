# kenkyu_project
### TSUBAME setup
#### Interactive node
qrsh -g tga-hpc-lecture -l f_node=2 -l h_rt=1:30:00

#### Modules
echo '' >> ~/.bashrc
echo '# Modules' >> ~/.bashrc
echo '. /etc/profile.d/modules.sh' >> ~/.bashrc
echo 'module load cuda openmpi nccl cudnn' >> ~/.bashrc
source ~/.bashrc

#### Install Pyenv Virtualenv
git clone https://github.com/yyuu/pyenv.git ~/.pyenv
git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
echo '' >> ~/.bash_profile
echo '# Pyenv Virtualenv' >> ~/.bash_profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
source ~/.bash_profile
pyenv install 3.8.6
pyenv virtualenv 3.8.6 pytorch

### Code
#### Clone
git clone https://github..com/rioyokotalab/kenkyu_project
#### Move to folder
cd kenkyu_project
#### Pip install
pip install -r requirements.txt
#### Run
python 00_numpy.py
mpirun -npernode 4 -np 8 python 12_distributed.py
#### Update code
git pull
#### WandB
wandb login
#### Sweep
wandb sweep sweep.yaml
