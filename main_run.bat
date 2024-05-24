set MyConfig=--epoch 20 --dataset mnist

python.exe main.py --gan_type GAN --gpu_mode False %MyConfig%
python.exe main.py --gan_type CGAN --gpu_mode False %MyConfig%
python.exe main.py --gan_type infoGAN --gpu_mode False %MyConfig%
python.exe main.py --gan_type WGAN_GP --gpu_mode False %MyConfig%

pause
exit


:: sec: ���L�͊m�F�L�^

set MyConfig=--epoch 20 --dataset mnist

:: ��1ep���琶������đ����A�m�F2405
python.exe main.py --gan_type GAN --gpu_mode False %MyConfig%
python.exe main.py --gan_type CGAN --gpu_mode False %MyConfig%
python.exe main.py --gan_type infoGAN --gpu_mode False %MyConfig%

:: ��14�`20ep�ł���Ɛ�������n�߂�A�x��
python.exe main.py --gan_type WGAN_GP --gpu_mode False %MyConfig%

:: --gan_type: [v]�͓���OK�m�F�ς�
:: 'GAN'[v], 'CGAN'[v], 'infoGAN'[v], 'ACGAN'[v], 'EBGAN'[v], 'BEGAN'[v], 'WGAN'[v], 'WGAN_GP'[v], 'DRAGAN'[v], 'LSGAN'[v]
:: --dataset: 
:: 'mnist'[v], 'fashion-mnist'[v], 'cifar10'[v], 'svhn', 'stl10', 'lsun-bed'

pause
