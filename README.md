# MWGAN
Optical Scattering Imaging Based on Multi-frame Wavelet Model

You can refer to the following commands to train the model.
python main.py  --exp_dir /home/yang/project/MWGAN/experiments_BD/MWGAN/001  --mode train --model TecoGAN --opt /home/yang/project/MWGAN/experiments_BD/MWGAN/001/train.yml

You can refer to the following commands to test the model.
python main.py  --exp_dir /home/yang/project/MWGAN/experiments_BD/MWGAN/001  --mode test --model TecoGAN --opt /home/yang/project/MWGAN/experiments_BD/MWGAN/001/test.yml
