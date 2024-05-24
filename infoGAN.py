import utils, torch, time, os, pickle, itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import dataloader

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, len_discrete_code=10, len_continuous_code=2):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.len_discrete_code + self.len_continuous_code, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input, cont_code, dist_code):
        x = torch.cat([input, cont_code, dist_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, len_discrete_code=10, len_continuous_code=2):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
            # nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)
        a = F.sigmoid(x[:, self.output_dim])
        b = x[:, self.output_dim:self.output_dim + self.len_continuous_code]
        c = x[:, self.output_dim + self.len_continuous_code:]

        return a, b, c

class infoGAN(object):
    def __init__(self, args, SUPERVISED=True):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.SUPERVISED = SUPERVISED        # if it is true, label info is directly used for code
        self.len_discrete_code = 10         # categorical distribution (i.e. label)
        self.len_continuous_code = 2        # gaussian distribution (e.g. rotation, thickness)
        self.sample_num = self.len_discrete_code ** 2

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size, len_discrete_code=self.len_discrete_code, len_continuous_code=self.len_continuous_code)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, len_discrete_code=self.len_discrete_code, len_continuous_code=self.len_continuous_code)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.info_optimizer = optim.Adam(itertools.chain(self.G.parameters(), self.D.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.MSE_loss = nn.MSELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        # ↓10x10の画像を生成するために、10x10個分のzベクトルを生成・ランダム値を代入
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.len_discrete_code):
            self.sample_z_[i * self.len_discrete_code] = torch.rand(1, self.z_dim)
            for j in range(1, self.len_discrete_code):
                self.sample_z_[i * self.len_discrete_code + j] = self.sample_z_[i * self.len_discrete_code]

        temp = torch.zeros((self.len_discrete_code, 1))
        for i in range(self.len_discrete_code):
            temp[i, 0] = i
        # ↑tempは、[0,1,2,...,9]
        
        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.len_discrete_code):
            temp_y[i * self.len_discrete_code: (i + 1) * self.len_discrete_code] = temp
        # ↑temp_yは、[0,1,2,...,9, 0,1,2,...,9, ... 0,1,2,...,9]

        # ↓テスト用の固定yベクトルは、[eye(10), eye(10), ... eye(10)]^T
        self.sample_y_ = torch.zeros((self.sample_num, self.len_discrete_code)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        # scatter_(dim, index, src) の操作では、dim は次元（この場合1、すなわち列方向）、index はインデックス（この場合temp_y）、src は代入する値（この場合は1）
        
        # ↓テスト用の固定cベクトルは、全ゼロのベクトルを使用している
        self.sample_c_ = torch.zeros((self.sample_num, self.len_continuous_code))

        # manipulating two continuous code
        self.sample_z2_ = torch.rand((1, self.z_dim)).expand(self.sample_num, self.z_dim)
        # ↓連続変化可視化用の固定yベクトルは、0番目のみ1のベクトルを使用
        self.sample_y2_ = torch.zeros(self.sample_num, self.len_discrete_code)
        self.sample_y2_[:, 0] = 1

        # ↓連続変化可視化用の固定cベクトルは、linspaceで-1～1を10分割した連続変化の値を使用
        temp_c = torch.linspace(-1, 1, 10)
        self.sample_c2_ = torch.zeros((self.sample_num, 2))
        for i in range(self.len_discrete_code):
            for j in range(self.len_discrete_code):
                self.sample_c2_[i*self.len_discrete_code+j, 0] = temp_c[i]
                self.sample_c2_[i*self.len_discrete_code+j, 1] = temp_c[j]

        if self.gpu_mode:
            self.sample_z_, self.sample_y_, self.sample_c_, self.sample_z2_, self.sample_y2_, self.sample_c2_ = \
                self.sample_z_.cuda(), self.sample_y_.cuda(), self.sample_c_.cuda(), self.sample_z2_.cuda(), \
                self.sample_y2_.cuda(), self.sample_c2_.cuda()
        
      # debug: テスト用の各固定ベクトルを出力
        import pathlib
        path_out = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/'
        torch.set_printoptions(profile="full") # PyTorchのテンソルを省略せずにテキストで完全に表示
        pathlib.Path(path_out + "debug sample_z_.txt").write_text(str(self.sample_z_))
        pathlib.Path(path_out + "debug sample_y_.txt").write_text(str(self.sample_y_))
        pathlib.Path(path_out + "debug sample_c_.txt").write_text(str(self.sample_c_))
        pathlib.Path(path_out + "debug sample_z2_.txt").write_text(str(self.sample_z2_))
        pathlib.Path(path_out + "debug sample_y2_.txt").write_text(str(self.sample_y2_))
        pathlib.Path(path_out + "debug sample_c2_.txt").write_text(str(self.sample_c2_))
        torch.set_printoptions(profile="default")
      # debug: end

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['info_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        # self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        # ↓
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size), torch.zeros(self.batch_size) # エラーが発生したため、サイズが一致するように修正2405 [tag=ERR1]
        
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.SUPERVISED == True:
                    y_disc_ = torch.zeros((self.batch_size, self.len_discrete_code)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                else:
                    # メモ: 既定で、SUPERVISED=Falseが実行される
                    # ラベル情報なしで自然に分類される！ランダムなラベル値の列を与えるのみ(下記)
                    y_disc_ = torch.from_numpy(
                        np.random.multinomial(1, self.len_discrete_code * [float(1.0 / self.len_discrete_code)],
                                              size=[self.batch_size])).type(torch.FloatTensor)

                # メモ: 教師なしで自然に調整可能な特徴が得られる！ランダムな連続値の列を与えるのみ(下記)
                y_cont_ = torch.from_numpy(np.random.uniform(-1, 1, size=(self.batch_size, 2))).type(torch.FloatTensor)

                if self.gpu_mode:
                    x_, z_, y_disc_, y_cont_ = x_.cuda(), z_.cuda(), y_disc_.cuda(), y_cont_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real, _, _ = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_) # エラー発生→修正2405 [tag=ERR1]

                G_ = self.G(z_, y_cont_, y_disc_)
                D_fake, _, _ = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_cont_, y_disc_)
                D_fake, D_cont, D_disc = self.D(G_)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                # G_loss.backward(retain_graph=True)
                # ↓
                G_loss.backward() # 下記で再計算するので不要 修正2405 [tag=ERR2]
                
                self.G_optimizer.step()

                # information loss

                # 下記: 上記の「D_optimizer.step()」「G_optimizer.step()」で更新されるので、
                # G・D両モデル内のパラメータ値が変わり、値・勾配の再利用は不可となり、
                # 下記の再計算が必要 追加2405 [tag=ERR2]
                G_ = self.G(z_, y_cont_, y_disc_)
                D_fake, D_cont, D_disc = self.D(G_)

                disc_loss = self.CE_loss(D_disc, torch.max(y_disc_, 1)[1])
                cont_loss = self.MSE_loss(D_cont, y_cont_)
                info_loss = disc_loss + cont_loss
                self.train_hist['info_loss'].append(info_loss.item())

                info_loss.backward() # エラー発生→修正2405 [tag=ERR2]
                # エラー内容: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                # コード内で+=, *=, .add_(), .mul_()などのin-place操作を見つけて、それらを非in-place版（例：.add(), .mul()）に置き換えます。これにより、元のデータが保持され、勾配計算時に必要な情報が失われません。
                self.info_optimizer.step()


                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, info_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item(), info_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_cont',
                                 self.epoch)
        self.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        """ style by class """
        # ↓毎回固定のz,c,yベクトルを与えて、テスト用の画像を生成する
        samples = self.G(self.sample_z_, self.sample_c_, self.sample_y_)
        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

        """ manipulating two continous codes """
        samples = self.G(self.sample_z2_, self.sample_c2_, self.sample_y2_)
        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_cont_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def loss_plot(self, hist, path='Train_hist.png', model_name=''):
        x = range(len(hist['D_loss']))

        y1 = hist['D_loss']
        y2 = hist['G_loss']
        y3 = hist['info_loss']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')
        plt.plot(x, y3, label='info_loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(path, model_name + '_loss.png')

        plt.savefig(path)