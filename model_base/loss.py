import torch
import torch.nn as nn
import torch.nn.functional as F

# Losses
l2_norm = nn.MSELoss(reduction='sum')
l1_loss = nn.L1Loss(reduction='sum')
Kld_loss = nn.KLDivLoss(reduction='sum')


def Distributed_Alignment_Loss_M(mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, logvar_x1, logvar_x2, logvar_x3, logvar_x4,logvar_x5):
    #   NOTE: USE COVARIANCE MATRIX. NOT STANDARD DEVIATION
    #   sigma_x = torch.exp(0.25*logvar_x)
    #   sigma_sig = torch.exp(0.25*logvar_sig)
    sigma_x1 = logvar_x1.exp()
    sigma_x2 = logvar_x2.exp()
    sigma_x3 = logvar_x3.exp()
    sigma_x4 = logvar_x4.exp()
    sigma_x5 = logvar_x5.exp()

    mu_loss_W1 = l2_norm(mu_x1, (1/4*(mu_x2 + mu_x3 + mu_x4 +mu_x5))) + torch.norm((sigma_x1 - (1/4*(sigma_x2 + sigma_x3 + sigma_x4 + sigma_x5))))
    mu_loss_W2 = l2_norm(mu_x2, (1/4*(mu_x1 + mu_x3 + mu_x4 +mu_x5))) + torch.norm((sigma_x2 - (1/4*(sigma_x1 + sigma_x3 + sigma_x4 + sigma_x5))))
    mu_loss_W3 = l2_norm(mu_x3, (1/4*(mu_x2 + mu_x1 + mu_x4 +mu_x5))) + torch.norm((sigma_x3 - (1/4*(sigma_x2 + sigma_x1 + sigma_x4 + sigma_x5))))
    mu_loss_W4 = l2_norm(mu_x4, (1/4*(mu_x2 + mu_x3 + mu_x1 +mu_x5))) + torch.norm((sigma_x4 - (1/4*(sigma_x2 + sigma_x3 + sigma_x1 + sigma_x5))))
    mu_loss_W5 = l2_norm(mu_x5, (1/4*(mu_x2 + mu_x3 + mu_x4 +mu_x1))) + torch.norm((sigma_x5 - (1/4*(sigma_x2 + sigma_x3 + sigma_x4 + sigma_x1))))

    da_loss = torch.sqrt(mu_loss_W1 + mu_loss_W2 + mu_loss_W3 + mu_loss_W4 + mu_loss_W5)
    return da_loss


def VAE_Loss_M(beta, recon_x1, x1, recon_x2, x2, recon_x3, x3, recon_x4, x4, recon_x5, x5,
             logvar_x1, logvar_x2, logvar_x3, logvar_x4, logvar_x5,
             mu_x1, mu_x2, mu_x3, mu_x4, mu_x5):
    reconstruction_1 = l1_loss(recon_x1, x1)
    reconstruction_2 = l1_loss(recon_x2, x2)
    reconstruction_3 = l1_loss(recon_x3, x3)
    reconstruction_4 = l1_loss(recon_x4, x4)
    reconstruction_5 = l1_loss(recon_x5, x5)

    KLD_1 = 0.5 * torch.sum(1 + logvar_x1 - mu_x1.pow(2) - logvar_x1.exp())
    KLD_2 = 0.5 * torch.sum(1 + logvar_x2 - mu_x2.pow(2) - logvar_x2.exp())
    KLD_3 = 0.5 * torch.sum(1 + logvar_x3 - mu_x3.pow(2) - logvar_x3.exp())
    KLD_4 = 0.5 * torch.sum(1 + logvar_x4 - mu_x4.pow(2) - logvar_x4.exp())
    KLD_5 = 0.5 * torch.sum(1 + logvar_x5 - mu_x5.pow(2) - logvar_x5.exp())

    vae_loss = reconstruction_1 + reconstruction_2 + reconstruction_3 + reconstruction_4 + reconstruction_5 - beta * (
                KLD_1 + KLD_2 + KLD_3 + KLD_4 + KLD_5)
    return vae_loss


def ZS_loss_M(x1, x2, x3, x4, x5, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5):
    reconstruction_11 = l1_loss(recon_s1, x1)
    reconstruction_22 = l1_loss(recon_s2, x2)
    reconstruction_33 = l1_loss(recon_s3, x3)
    reconstruction_44 = l1_loss(recon_s4, x4)
    reconstruction_55 = l1_loss(recon_s5, x5)
    zs_loss = reconstruction_11 + reconstruction_22 + reconstruction_33 + reconstruction_44 + reconstruction_55
    return zs_loss


def loss_function_M(x1, x2, x3, x4, x5, recon_x1, recon_x2, recon_x3, recon_x4, recon_x5,
                  mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, logvar_x1, logvar_x2, logvar_x3, logvar_x4, logvar_x5,
                  beta, delta,  z_s, alpha, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5):
    da_loss = Distributed_Alignment_Loss_M(mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, logvar_x1, logvar_x2, logvar_x3, logvar_x4,
                                         logvar_x5)
    vae_loss = VAE_Loss_M(beta, recon_x1, x1, recon_x2, x2, recon_x3, x3, recon_x4, x4, recon_x5, x5, logvar_x1,
                        logvar_x2, logvar_x3, logvar_x4, logvar_x5, mu_x1, mu_x2, mu_x3, mu_x4, mu_x5)
    zs_loss = ZS_loss_M(x1, x2, x3, x4, x5, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5)

    cada_vae_loss = vae_loss + delta * da_loss + alpha * zs_loss
    return cada_vae_loss


def Distributed_Alignment_Loss_Y(mu_x1, mu_x2, mu_x3, logvar_x1, logvar_x2, logvar_x3):
    #   NOTE: USE COVARIANCE MATRIX. NOT STANDARD DEVIATION
    #   sigma_x = torch.exp(0.25*logvar_x)
    #   sigma_sig = torch.exp(0.25*logvar_sig)
    sigma_x1 = logvar_x1.exp()
    sigma_x2 = logvar_x2.exp()
    sigma_x3 = logvar_x3.exp()

    mu_loss_W1 = l2_norm(mu_x1, (1 / 2 * (mu_x2 + mu_x3))) + torch.norm(
        (sigma_x1 - (1 / 2 * (sigma_x2 + sigma_x3))))
    mu_loss_W2 = l2_norm(mu_x2, (1 / 2 * (mu_x1 + mu_x3))) + torch.norm(
        (sigma_x2 - (1 / 2 * (sigma_x1 + sigma_x3))))
    mu_loss_W3 = l2_norm(mu_x3, (1 / 2 * (mu_x2 + mu_x1))) + torch.norm(
        (sigma_x3 - (1 / 2 * (sigma_x2 + sigma_x1))))

    da_loss = torch.sqrt(mu_loss_W1 + mu_loss_W2 + mu_loss_W3)
    # da_loss.sum()
    return da_loss

def VAE_Loss_Y(beta, recon_x1, x1, recon_x2, x2, recon_x3, x3, logvar_x1, logvar_x2,logvar_x3, mu_x1, mu_x2, mu_x3):
    reconstruction_1 = l1_loss(recon_x1, x1)
    reconstruction_2 = l1_loss(recon_x2, x2)
    reconstruction_3 = l1_loss(recon_x3, x3)

    KLD_1 = 0.5 * torch.sum(1 + logvar_x1 - mu_x1.pow(2) - logvar_x1.exp())
    KLD_2 = 0.5 * torch.sum(1 + logvar_x2 - mu_x2.pow(2) - logvar_x2.exp())
    KLD_3 = 0.5 * torch.sum(1 + logvar_x3 - mu_x3.pow(2) - logvar_x3.exp())

    vae_loss = reconstruction_1 + reconstruction_2 + reconstruction_3  - beta * (KLD_1 + KLD_2 + KLD_3)
    return vae_loss


def ZS_loss_Y(x1, x2, x3, recon_s1, recon_s2, recon_s3):
    reconstruction_11 = l1_loss(recon_s1, x1)
    reconstruction_22 = l1_loss(recon_s2, x2)
    reconstruction_33 = l1_loss(recon_s3, x3)
    zs_loss = reconstruction_11 + reconstruction_22 + reconstruction_33
    return zs_loss


def loss_function_Y(x1, x2, x3, recon_x1, recon_x2, recon_x3,
                  mu_x1, mu_x2, mu_x3, logvar_x1, logvar_x2, logvar_x3,
                  beta, delta, alpha, recon_s1, recon_s2,recon_s3):
    da_loss = Distributed_Alignment_Loss_Y(mu_x1, mu_x2, mu_x3, logvar_x1, logvar_x2, logvar_x3)
    vae_loss = VAE_Loss_Y(beta, recon_x1, x1, recon_x2, x2, recon_x3, x3, logvar_x1, logvar_x2,logvar_x3, mu_x1, mu_x2, mu_x3)
    zs_loss = ZS_loss_Y(x1, x2, x3, recon_s1, recon_s2, recon_s3)

    cada_vae_loss = vae_loss + delta * da_loss + alpha * zs_loss
    return cada_vae_loss

def Distributed_Alignment_Loss_N(mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, logvar_x1, logvar_x2, logvar_x3, logvar_x4,logvar_x5):
    #   NOTE: USE COVARIANCE MATRIX. NOT STANDARD DEVIATION
    #   sigma_x = torch.exp(0.25*logvar_x)
    #   sigma_sig = torch.exp(0.25*logvar_sig)
    sigma_x1 = logvar_x1.exp()
    sigma_x2 = logvar_x2.exp()
    sigma_x3 = logvar_x3.exp()
    sigma_x4 = logvar_x4.exp()
    sigma_x5 = logvar_x5.exp()

    mu_loss_W1 = l2_norm(mu_x1, (1/4*(mu_x2 + mu_x3 + mu_x4 +mu_x5))) + torch.norm((sigma_x1 - (1/4*(sigma_x2 + sigma_x3 + sigma_x4 + sigma_x5))))
    mu_loss_W2 = l2_norm(mu_x2, (1/4*(mu_x1 + mu_x3 + mu_x4 +mu_x5))) + torch.norm((sigma_x2 - (1/4*(sigma_x1 + sigma_x3 + sigma_x4 + sigma_x5))))
    mu_loss_W3 = l2_norm(mu_x3, (1/4*(mu_x2 + mu_x1 + mu_x4 +mu_x5))) + torch.norm((sigma_x3 - (1/4*(sigma_x2 + sigma_x1 + sigma_x4 + sigma_x5))))
    mu_loss_W4 = l2_norm(mu_x4, (1/4*(mu_x2 + mu_x3 + mu_x1 +mu_x5))) + torch.norm((sigma_x4 - (1/4*(sigma_x2 + sigma_x3 + sigma_x1 + sigma_x5))))
    mu_loss_W5 = l2_norm(mu_x5, (1/4*(mu_x2 + mu_x3 + mu_x4 +mu_x1))) + torch.norm((sigma_x5 - (1/4*(sigma_x2 + sigma_x3 + sigma_x4 + sigma_x1))))

    da_loss = torch.sqrt(mu_loss_W1 + mu_loss_W2 + mu_loss_W3 + mu_loss_W4 + mu_loss_W5)
    # da_loss.sum()
    return da_loss


def VAE_Loss_N(beta, recon_x1, x1, recon_x2, x2, recon_x3, x3, recon_x4, x4, recon_x5, x5,
             logvar_x1, logvar_x2, logvar_x3, logvar_x4, logvar_x5,
             mu_x1, mu_x2, mu_x3, mu_x4, mu_x5):
    reconstruction_1 = l1_loss(recon_x1, x1)
    reconstruction_2 = l1_loss(recon_x2, x2)
    reconstruction_3 = l1_loss(recon_x3, x3)
    reconstruction_4 = l1_loss(recon_x4, x4)
    reconstruction_5 = l1_loss(recon_x5, x5)

    KLD_1 = 0.5 * torch.sum(1 + logvar_x1 - mu_x1.pow(2) - logvar_x1.exp())
    KLD_2 = 0.5 * torch.sum(1 + logvar_x2 - mu_x2.pow(2) - logvar_x2.exp())
    KLD_3 = 0.5 * torch.sum(1 + logvar_x3 - mu_x3.pow(2) - logvar_x3.exp())
    KLD_4 = 0.5 * torch.sum(1 + logvar_x4 - mu_x4.pow(2) - logvar_x4.exp())
    KLD_5 = 0.5 * torch.sum(1 + logvar_x5 - mu_x5.pow(2) - logvar_x5.exp())

    vae_loss = reconstruction_1 + reconstruction_2 + reconstruction_3 + reconstruction_4 + reconstruction_5 - beta * (
                KLD_1 + KLD_2 + KLD_3 + KLD_4 + KLD_5)
    return vae_loss


def ZS_loss_N(x1, x2, x3, x4, x5, z_s, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5):
    reconstruction_11 = l1_loss(recon_s1, x1)
    reconstruction_22 = l1_loss(recon_s2, x2)
    reconstruction_33 = l1_loss(recon_s3, x3)
    reconstruction_44 = l1_loss(recon_s4, x4)
    reconstruction_55 = l1_loss(recon_s5, x5)
    zs_loss = reconstruction_11 + reconstruction_22 + reconstruction_33 + reconstruction_44 + reconstruction_55
    return zs_loss


def loss_function_N(x1, x2, x3, x4, x5, recon_x1, recon_x2, recon_x3, recon_x4, recon_x5,
                  mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, logvar_x1, logvar_x2, logvar_x3, logvar_x4, logvar_x5,
                    beta, delta, z_s, alpha, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5):

    da_loss = Distributed_Alignment_Loss_N(mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, logvar_x1, logvar_x2, logvar_x3, logvar_x4,
                                         logvar_x5)
    vae_loss = VAE_Loss_N(beta, recon_x1, x1, recon_x2, x2, recon_x3, x3, recon_x4, x4, recon_x5, x5, logvar_x1,
                        logvar_x2, logvar_x3, logvar_x4, logvar_x5, mu_x1, mu_x2, mu_x3, mu_x4, mu_x5)
    zs_loss = ZS_loss_N(x1, x2, x3, x4, x5, z_s, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5)

    cada_vae_loss = vae_loss + delta * da_loss + alpha * zs_loss

    return cada_vae_loss

def Distributed_Alignment_Loss_A(mu_x1, mu_x2, mu_x3, mu_x4, logvar_x1, logvar_x2, logvar_x3, logvar_x4):
    #   NOTE: USE COVARIANCE MATRIX. NOT STANDARD DEVIATION
    #   sigma_x = torch.exp(0.25*logvar_x)
    #   sigma_sig = torch.exp(0.25*logvar_sig)
    sigma_x1 = logvar_x1.exp()
    sigma_x2 = logvar_x2.exp()
    sigma_x3 = logvar_x3.exp()
    sigma_x4 = logvar_x4.exp()

    mu_loss_W1 = l2_norm(mu_x1, (1 / 3 * (mu_x2 + mu_x3 + mu_x4))) + \
                 torch.norm((sigma_x1 - (1 / 3 * (sigma_x2 + sigma_x3 + sigma_x4))))
    mu_loss_W2 = l2_norm(mu_x2, (1 / 3 * (mu_x1 + mu_x3 + mu_x4))) + \
                 torch.norm((sigma_x2 - (1 / 3 * (sigma_x1 + sigma_x3 + sigma_x4))))
    mu_loss_W3 = l2_norm(mu_x3, (1 / 3 * (mu_x2 + mu_x1 + mu_x4))) + \
                 torch.norm((sigma_x3 - (1 / 3 * (sigma_x2 + sigma_x1 + sigma_x4))))
    mu_loss_W4 = l2_norm(mu_x4, (1 / 3 * (mu_x2 + mu_x3 + mu_x1))) + \
                 torch.norm((sigma_x4 - (1 / 3 * (sigma_x2 + sigma_x3 + sigma_x1))))

    da_loss = torch.sqrt(mu_loss_W1 + mu_loss_W2 + mu_loss_W3 + mu_loss_W4)
    # da_loss.sum()
    return da_loss


def VAE_Loss_A(beta, recon_x1, x1, recon_x2, x2, recon_x3, x3, recon_x4, x4,
             logvar_x1, logvar_x2, logvar_x3, logvar_x4,
             mu_x1, mu_x2, mu_x3, mu_x4):
    reconstruction_1 = l1_loss(recon_x1, x1)
    reconstruction_2 = l1_loss(recon_x2, x2)
    reconstruction_3 = l1_loss(recon_x3, x3)
    reconstruction_4 = l1_loss(recon_x4, x4)

    KLD_1 = 0.5 * torch.sum(1 + logvar_x1 - mu_x1.pow(2) - logvar_x1.exp())
    KLD_2 = 0.5 * torch.sum(1 + logvar_x2 - mu_x2.pow(2) - logvar_x2.exp())
    KLD_3 = 0.5 * torch.sum(1 + logvar_x3 - mu_x3.pow(2) - logvar_x3.exp())
    KLD_4 = 0.5 * torch.sum(1 + logvar_x4 - mu_x4.pow(2) - logvar_x4.exp())

    vae_loss = reconstruction_1 + reconstruction_2 + reconstruction_3 + reconstruction_4 \
               - beta * (KLD_1 + KLD_2 + KLD_3 + KLD_4)
    #   print("LOSS VAE:", reconstruction_1.item(),reconstruction_2.item(), KLD_1.item(), KLD_2.item())
    return vae_loss


def ZS_loss_A(x1, x2, x3, x4, z_s, recon_s1, recon_s2, recon_s3, recon_s4):
    reconstruction_11 = l1_loss(recon_s1, x1)
    reconstruction_22 = l1_loss(recon_s2, x2)
    reconstruction_33 = l1_loss(recon_s3, x3)
    reconstruction_44 = l1_loss(recon_s4, x4)
    zs_loss = reconstruction_11 + reconstruction_22 + reconstruction_33 + reconstruction_44
    return zs_loss


def loss_function_A(x1, x2, x3, x4, recon_x1, recon_x2, recon_x3, recon_x4,
                  mu_x1, mu_x2, mu_x3, mu_x4,
                  logvar_x1, logvar_x2, logvar_x3, logvar_x4,
                  beta, delta, z_s, alpha, recon_s1, recon_s2, recon_s3,recon_s4):
    da_loss = Distributed_Alignment_Loss_A(mu_x1, mu_x2, mu_x3, mu_x4, logvar_x1, logvar_x2, logvar_x3, logvar_x4)
    vae_loss = VAE_Loss_A(beta, recon_x1, x1, recon_x2, x2, recon_x3, x3, recon_x4, x4,
                        logvar_x1, logvar_x2, logvar_x3, logvar_x4,
                        mu_x1, mu_x2, mu_x3, mu_x4)
    zs_loss = ZS_loss_A(x1, x2, x3, x4, z_s, recon_s1, recon_s2, recon_s3, recon_s4)

    cada_vae_loss = vae_loss + delta * da_loss + alpha * zs_loss

    return cada_vae_loss

def Distributed_Alignment_Loss_C(mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6,
                               logvar_x1, logvar_x2, logvar_x3, logvar_x4, logvar_x5, logvar_x6):
    #   NOTE: USE COVARIANCE MATRIX. NOT STANDARD DEVIATION
    #   sigma_x = torch.exp(0.25*logvar_x)
    #   sigma_sig = torch.exp(0.25*logvar_sig)
    sigma_x1 = logvar_x1.exp()
    sigma_x2 = logvar_x2.exp()
    sigma_x3 = logvar_x3.exp()
    sigma_x4 = logvar_x4.exp()
    sigma_x5 = logvar_x5.exp()
    sigma_x6 = logvar_x6.exp()

    mu_loss_W1 = l2_norm(mu_x1, (1 / 5 * (mu_x2 + mu_x3 + mu_x4 + mu_x5 + mu_x6))) + \
                 torch.norm((sigma_x1 - (1 / 5 * (sigma_x2 + sigma_x3 + sigma_x4 + sigma_x5 + sigma_x6))))
    mu_loss_W2 = l2_norm(mu_x2, (1 / 5 * (mu_x1 + mu_x3 + mu_x4 + mu_x5 + mu_x6))) + \
                 torch.norm((sigma_x2 - (1 / 5 * (sigma_x1 + sigma_x3 + sigma_x4 + sigma_x5 + sigma_x6))))
    mu_loss_W3 = l2_norm(mu_x3, (1 / 5 * (mu_x2 + mu_x1 + mu_x4 + mu_x5 + mu_x6))) + \
                 torch.norm((sigma_x3 - (1 / 5 * (sigma_x2 + sigma_x1 + sigma_x4 + sigma_x5 + sigma_x6))))
    mu_loss_W4 = l2_norm(mu_x4, (1 / 5 * (mu_x2 + mu_x3 + mu_x1 + mu_x5 + mu_x6))) + \
                 torch.norm((sigma_x4 - (1 / 5 * (sigma_x2 + sigma_x3 + sigma_x1 + sigma_x5 + sigma_x6))))
    mu_loss_W5 = l2_norm(mu_x5, (1 / 5 * (mu_x2 + mu_x3 + mu_x4 + mu_x1 + mu_x6))) + \
                 torch.norm((sigma_x5 - (1 / 5 * (sigma_x2 + sigma_x3 + sigma_x4 + sigma_x1 + sigma_x6))))
    mu_loss_W6 = l2_norm(mu_x6, (1 / 5 * (mu_x2 + mu_x3 + mu_x4 + mu_x1 + mu_x5))) + \
                 torch.norm((sigma_x6 - (1 / 5 * (sigma_x2 + sigma_x3 + sigma_x4 + sigma_x1 + sigma_x5))))

    da_loss = torch.sqrt(mu_loss_W1 + mu_loss_W2 + mu_loss_W3 + mu_loss_W4 + mu_loss_W5 + mu_loss_W6)
    # da_loss.sum()
    return da_loss


def VAE_Loss_C(beta, recon_x1, x1, recon_x2, x2, recon_x3, x3, recon_x4, x4, recon_x5, x5, recon_x6, x6,
             logvar_x1, logvar_x2, logvar_x3, logvar_x4, logvar_x5, logvar_x6,
             mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6):
    reconstruction_1 = l1_loss(recon_x1, x1)
    reconstruction_2 = l1_loss(recon_x2, x2)
    reconstruction_3 = l1_loss(recon_x3, x3)
    reconstruction_4 = l1_loss(recon_x4, x4)
    reconstruction_5 = l1_loss(recon_x5, x5)
    reconstruction_6 = l1_loss(recon_x6, x6)

    KLD_1 = 0.5 * torch.sum(1 + logvar_x1 - mu_x1.pow(2) - logvar_x1.exp())
    KLD_2 = 0.5 * torch.sum(1 + logvar_x2 - mu_x2.pow(2) - logvar_x2.exp())
    KLD_3 = 0.5 * torch.sum(1 + logvar_x3 - mu_x3.pow(2) - logvar_x3.exp())
    KLD_4 = 0.5 * torch.sum(1 + logvar_x4 - mu_x4.pow(2) - logvar_x4.exp())
    KLD_5 = 0.5 * torch.sum(1 + logvar_x5 - mu_x5.pow(2) - logvar_x5.exp())
    KLD_6 = 0.5 * torch.sum(1 + logvar_x6 - mu_x6.pow(2) - logvar_x6.exp())

    vae_loss = reconstruction_1 + reconstruction_2 + reconstruction_3 + reconstruction_4 + reconstruction_5 + \
               reconstruction_6 - beta * (KLD_1 + KLD_2 + KLD_3 + KLD_4 + KLD_5 + KLD_6)
    #   print("LOSS VAE:", reconstruction_1.item(),reconstruction_2.item(), KLD_1.item(), KLD_2.item())
    return vae_loss


def ZS_loss_C(x1, x2, x3, x4, x5, x6, z_s, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5, recon_s6):
    reconstruction_11 = l1_loss(recon_s1, x1)
    reconstruction_22 = l1_loss(recon_s2, x2)
    reconstruction_33 = l1_loss(recon_s3, x3)
    reconstruction_44 = l1_loss(recon_s4, x4)
    reconstruction_55 = l1_loss(recon_s5, x5)
    reconstruction_66 = l1_loss(recon_s6, x6)
    zs_loss = reconstruction_11 + reconstruction_22 + reconstruction_33 + reconstruction_44 + reconstruction_55 + reconstruction_66
    return zs_loss


def loss_function_C(x1, x2, x3, x4, x5, x6, recon_x1, recon_x2, recon_x3, recon_x4, recon_x5, recon_x6,
                  mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6,
                  logvar_x1, logvar_x2, logvar_x3, logvar_x4, logvar_x5, logvar_x6,
                  beta, delta, z_s, alpha,  recon_s1, recon_s2, recon_s3, recon_s4, recon_s5, recon_s6):

    da_loss = Distributed_Alignment_Loss_C(mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6,
                                         logvar_x1, logvar_x2, logvar_x3, logvar_x4, logvar_x5, logvar_x6)
    vae_loss = VAE_Loss_C(beta, recon_x1, x1, recon_x2, x2, recon_x3, x3, recon_x4, x4, recon_x5, x5, recon_x6, x6,
                        logvar_x1, logvar_x2, logvar_x3, logvar_x4, logvar_x5, logvar_x6,
                        mu_x1, mu_x2, mu_x3, mu_x4, mu_x5, mu_x6)
    zs_loss = ZS_loss_C(x1, x2, x3, x4, x5, x6, z_s, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5, recon_s6)

    cada_vae_loss = vae_loss + delta * da_loss + alpha * zs_loss

    return cada_vae_loss


def loss_func(feat,cluster_centers):
    alpha = 1.0
    q = 1.0 / (1.0 + torch.sum((feat.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()

    log_q = torch.log(q)
    loss = F.kl_div(log_q, p)
    return loss, p


def dist_2_label(q_t):
    _, label = torch.max(q_t, dim=1)
    return label.data.cpu().numpy()

