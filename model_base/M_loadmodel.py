import torch
import torch.nn as nn

class encoder_cada(nn.Module):
    def __init__(self, input_dim=[], hidden_dim=[500, 500, 1024], z=10):
        super(encoder_cada, self).__init__()
        self.encoder1 = nn.Sequential(nn.Linear(input_dim[0],hidden_dim[0]),nn.ReLU(),
                                         nn.Linear(hidden_dim[0],hidden_dim[1]),nn.ReLU(),
                                         nn.Linear(hidden_dim[1],hidden_dim[2]),nn.ReLU())
        self.mu1 = nn.Linear(hidden_dim[2], z)
        self.logvar1 = nn.Linear(hidden_dim[2], z)

        self.encoder2 = nn.Sequential(nn.Linear(input_dim[1],hidden_dim[0]),nn.ReLU(),
                                         nn.Linear(hidden_dim[0],hidden_dim[1]),nn.ReLU(),
                                         nn.Linear(hidden_dim[1],hidden_dim[2]),nn.ReLU())
        self.mu2 = nn.Linear(hidden_dim[2], z)
        self.logvar2 = nn.Linear(hidden_dim[2], z)

        self.encoder3 = nn.Sequential(nn.Linear(input_dim[2],hidden_dim[0]),nn.ReLU(),
                                         nn.Linear(hidden_dim[0],hidden_dim[1]),nn.ReLU(),
                                         nn.Linear(hidden_dim[1],hidden_dim[2]),nn.ReLU())
        self.mu3 = nn.Linear(hidden_dim[2], z)
        self.logvar3 = nn.Linear(hidden_dim[2], z)

        self.encoder4 = nn.Sequential(nn.Linear(input_dim[3],hidden_dim[0]),nn.ReLU(),
                                         nn.Linear(hidden_dim[0],hidden_dim[1]),nn.ReLU(),
                                         nn.Linear(hidden_dim[1],hidden_dim[2]),nn.ReLU())
        self.mu4 = nn.Linear(hidden_dim[2], z)
        self.logvar4 = nn.Linear(hidden_dim[2], z)

        self.encoder5 = nn.Sequential(nn.Linear(input_dim[4],hidden_dim[0]),nn.ReLU(),
                                         nn.Linear(hidden_dim[0],hidden_dim[1]),nn.ReLU(),
                                         nn.Linear(hidden_dim[1],hidden_dim[2]),nn.ReLU())
        self.mu5 = nn.Linear(hidden_dim[2], z)
        self.logvar5 = nn.Linear(hidden_dim[2], z)

        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_5 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.fuse_weight_1.data.fill_(1 / 5)
        self.fuse_weight_2.data.fill_(1 / 5)
        self.fuse_weight_3.data.fill_(1 / 5)
        self.fuse_weight_4.data.fill_(1 / 5)
        self.fuse_weight_5.data.fill_(1 / 5)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)  # mean 0, std
        return eps.mul(std).add_(mu)

    def forward(self, inputs=[]):
        # VAE for feature vector
        x1 = self.encoder1(inputs[0])
        mu1 = self.mu1(x1)
        logvar1 = self.logvar1(x1)
        z_x1 = self.reparameterize(mu1, logvar1)

        x2 = self.encoder2(inputs[1])
        mu2 = self.mu2(x2)
        logvar2 = self.logvar2(x2)
        z_x2 = self.reparameterize(mu2, logvar2)

        x3 = self.encoder3(inputs[2])
        mu3 = self.mu3(x3)
        logvar3 = self.logvar3(x3)
        z_x3 = self.reparameterize(mu3, logvar3)

        x4 = self.encoder4(inputs[3])
        mu4 = self.mu4(x4)
        logvar4 = self.logvar4(x4)
        z_x4 = self.reparameterize(mu4, logvar4)

        x5 = self.encoder5(inputs[4])
        mu5 = self.mu5(x5)
        logvar5 = self.logvar5(x5)
        z_x5 = self.reparameterize(mu5, logvar5)

        z_s = torch.mul(self.fuse_weight_1, z_x1) + torch.mul(self.fuse_weight_2, z_x2) + \
              torch.mul(self.fuse_weight_3, z_x3) + torch.mul(self.fuse_weight_4, z_x4) + \
              torch.mul(self.fuse_weight_5, z_x5)

        return z_x1, z_x2, z_x3, z_x4, z_x5, \
               mu1, mu2, mu3, mu4, mu5, logvar1, logvar2, logvar3, logvar4, logvar5, z_s


class decoder_cada(nn.Module):
    def __init__(self, input_dim=[], hidden_dim=[500, 1024, 500], z=10):
        super(decoder_cada, self).__init__()
        self.decoder1 = nn.Sequential(nn.Linear(z,hidden_dim[2]),nn.ReLU(),
                                      nn.Linear(hidden_dim[2],hidden_dim[1]),nn.ReLU(),
                                      nn.Linear(hidden_dim[1],hidden_dim[0]),nn.ReLU(),
                                      nn.Linear(hidden_dim[0],input_dim[0]),nn.Sigmoid())

        self.decoder2 = nn.Sequential(nn.Linear(z,hidden_dim[2]),nn.ReLU(),
                                      nn.Linear(hidden_dim[2],hidden_dim[1]),nn.ReLU(),
                                      nn.Linear(hidden_dim[1],hidden_dim[0]),nn.ReLU(),
                                      nn.Linear(hidden_dim[0],input_dim[1]),nn.Sigmoid())

        self.decoder3 = nn.Sequential(nn.Linear(z,hidden_dim[2]),nn.ReLU(),
                                      nn.Linear(hidden_dim[2],hidden_dim[1]),nn.ReLU(),
                                      nn.Linear(hidden_dim[1],hidden_dim[0]),nn.ReLU(),
                                      nn.Linear(hidden_dim[0],input_dim[2]),nn.Sigmoid())

        self.decoder4 = nn.Sequential(nn.Linear(z,hidden_dim[2]),nn.ReLU(),
                                      nn.Linear(hidden_dim[2],hidden_dim[1]),nn.ReLU(),
                                      nn.Linear(hidden_dim[1],hidden_dim[0]),nn.ReLU(),
                                      nn.Linear(hidden_dim[0],input_dim[3]),nn.Sigmoid())

        self.decoder5 = nn.Sequential(nn.Linear(z,hidden_dim[2]),nn.ReLU(),
                                      nn.Linear(hidden_dim[2],hidden_dim[1]),nn.ReLU(),
                                      nn.Linear(hidden_dim[1],hidden_dim[0]),nn.ReLU(),
                                      nn.Linear(hidden_dim[0],input_dim[4]),nn.Sigmoid())

    def forward(self, z_x1, z_x2, z_x3, z_x4, z_x5, z_s):
        # VAE for feature vector
        recon_x1 = self.decoder1(z_x1)
        recon_x2 = self.decoder2(z_x2)
        recon_x3 = self.decoder3(z_x3)
        recon_x4 = self.decoder4(z_x4)
        recon_x5 = self.decoder5(z_x5)

        recon_s1 = self.decoder1(z_s)
        recon_s2 = self.decoder2(z_s)
        recon_s3 = self.decoder3(z_s)
        recon_s4 = self.decoder4(z_s)
        recon_s5 = self.decoder5(z_s)

        return recon_x1,recon_x2,recon_x3,recon_x4,recon_x5, \
               recon_s1,recon_s2,recon_s3,recon_s4,recon_s5