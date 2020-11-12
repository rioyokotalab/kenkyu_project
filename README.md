# kenkyu_project
### TSUBAME setup
#### Modules
echo '' >> ~/.bashrc
echo '# Modules' >> ~/.bashrc
echo '. /etc/profile.d/modules.sh' >> ~/.bashrc
echo 'module load cuda openmpi nccl cudnn' >> ~/.bashrc

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

### How to run

