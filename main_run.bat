set MyConfig=--epoch 20 --dataset mnist

python.exe main.py --gan_type GAN --gpu_mode False %MyConfig%
python.exe main.py --gan_type CGAN --gpu_mode False %MyConfig%
python.exe main.py --gan_type infoGAN --gpu_mode False %MyConfig%
python.exe main.py --gan_type WGAN_GP --gpu_mode False %MyConfig%

pause
exit


:: sec: 下記は確認記録

set MyConfig=--epoch 20 --dataset mnist

:: ↓1epから生成されて速い、確認2405
python.exe main.py --gan_type GAN --gpu_mode False %MyConfig%
python.exe main.py --gan_type CGAN --gpu_mode False %MyConfig%
python.exe main.py --gan_type infoGAN --gpu_mode False %MyConfig%

:: ↓14～20epでやっと生成され始める、遅い
python.exe main.py --gan_type WGAN_GP --gpu_mode False %MyConfig%

:: --gan_type: [v]は動作OK確認済み
:: 'GAN'[v], 'CGAN'[v], 'infoGAN'[v], 'ACGAN'[v], 'EBGAN'[v], 'BEGAN'[v], 'WGAN'[v], 'WGAN_GP'[v], 'DRAGAN'[v], 'LSGAN'[v]
:: --dataset: 
:: 'mnist'[v], 'fashion-mnist'[v], 'cifar10'[v], 'svhn', 'stl10', 'lsun-bed'

pause
