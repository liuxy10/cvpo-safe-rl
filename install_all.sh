# setup safety gym
cd envs/safety-gym
bash setup_mujoco.sh
source ~/.bashrc
python -m pip install -e .
cd ../..
python -m pip install -r requirement.txt
python -m pip install -e .

echo "********************************************************"
echo "********************************************************"
echo "********************************************************"
echo "                                                        "
echo "Please install pytorch manually to finish the env setup."
echo "                                                        "
echo "********************************************************"
echo "********************************************************"
echo "********************************************************"