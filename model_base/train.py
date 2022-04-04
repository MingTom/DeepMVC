import torchvision
from modeldatabase.model import *
import torch
import torch.optim as optim
from tqdm import tqdm
import joblib
import metrics
from time import sleep
from modeldatabase.loss import *
from metrics import *
from torch.autograd import Variable
from sklearn.cluster import KMeans

device = torch.device("cuda:0")
list_set = ['fuse_weight_1', 'fuse_weight_2', 'fuse_weight_3', 'fuse_weight_4', 'fuse_weight_5', 'fuse_weight_6']

def train(epoch,input_dim,hidden_dim,z,trainloader,inputs, Y, classes, data_name, beta=0,delta=0,alpha=10):
    model_cluster = KMeans(n_clusters=classes)
    if data_name == 'MSRCv1':
        encoder = encoder_cada_M(input_dim=input_dim, hidden_dim=hidden_dim, z=z).to(device)
        decoder = decoder_cada_M(input_dim=input_dim, hidden_dim=hidden_dim, z=z).to(device)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.00015, amsgrad=True)
        for i in range(epoch):
            print('epoch',i)
            trainbar = tqdm(trainloader)
            encoder.train()
            decoder.train()
            train_loss = 0
            if i > 5 and i < 21:
                delta += 0.54  # da_loss
            if i < 91:
                beta += 0.0026

            for batch_idx, (x1, y, x2, x3, x4, x5) in enumerate(trainbar):
                x1 = x1.float()  # print(x.size())  (64,3180)
                x2 = x2.float()
                x3 = x3.float()
                x4 = x4.float()
                x5 = x5.float()
                inputs_batchsize = [x1,x2,x3,x4,x5]
                z_x1, z_x2, z_x3, z_x4, z_x5, mu1, mu2, mu3, mu4, mu5, \
                logvar1, logvar2, logvar3, logvar4, logvar5, z_s = encoder(inputs_batchsize)

                recon_x1, recon_x2, recon_x3, recon_x4, recon_x5, \
                recon_s1, recon_s2, recon_s3, recon_s4, recon_s5 = decoder(z_x1,z_x2,z_x3,z_x4,z_x5,z_s)

                # loss
                loss = loss_function_M(x1, x2, x3, x4, x5, recon_x1, recon_x2, recon_x3, recon_x4, recon_x5,
                                     mu1, mu2, mu3, mu4, mu5, logvar1, logvar2, logvar3, logvar4, logvar5,
                                     beta, delta, z_s, alpha, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                all_loss = train_loss / (batch_idx + 1)
                trainbar.set_description('l:%.3f' % (all_loss))
                sleep(0.05)

    if data_name == 'Yale':
        encoder = encoder_cada_Y(input_dim=input_dim, hidden_dim=hidden_dim, z=z).to(device)
        decoder = decoder_cada_Y(input_dim=input_dim, hidden_dim=hidden_dim, z=z).to(device)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.00015, amsgrad=True)
        for i in range(epoch):
            print('epoch',i)
            trainbar = tqdm(trainloader)
            encoder.train()
            decoder.train()
            train_loss = 0
            if i > 5 and i < 21:
                delta += 0.54  # da_loss
            if i < 91:
                beta += 0.0026
            for batch_idx, (x1, y, x2, x3) in enumerate(trainbar):
                x1 = x1.float()  # print(x.size())  (64,3180)
                x2 = x2.float()
                x3 = x3.float()
                inputs_batchsize = [x1, x2, x3]
                z_x1, z_x2, z_x3, mu1, mu2, mu3, \
                logvar1, logvar2, logvar3, z_s = encoder(inputs_batchsize)

                recon_x1, recon_x2, recon_x3, \
                recon_s1, recon_s2, recon_s3 = decoder(z_x1, z_x2, z_x3, z_s)

                # loss
                loss = loss_function_Y(x1, x2, x3, recon_x1, recon_x2, recon_x3,
                                     mu1, mu2, mu3, logvar1, logvar2, logvar3,
                                     beta, delta, alpha, recon_s1, recon_s2, recon_s3)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                all_loss = train_loss / (batch_idx + 1)
                trainbar.set_description('l:%.3f' % (all_loss))
                sleep(0.05)

    if data_name == 'NUS':
        encoder = encoder_cada_N(input_dim=input_dim, hidden_dim=hidden_dim, z=z).to(device)
        decoder = decoder_cada_N(input_dim=input_dim, hidden_dim=hidden_dim, z=z).to(device)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.00015, amsgrad=True)
        for i in range(epoch):
            print('epoch',i)
            trainbar = tqdm(trainloader)
            encoder.train()
            decoder.train()
            train_loss = 0
            if i > 5 and i < 21:
                delta += 0.54  # da_loss
            if i < 91:
                beta += 0.0026

            for batch_idx, (x1, y, x2, x3, x4, x5) in enumerate(trainbar):
                x1 = x1.float()  # print(x.size())  (64,3180)
                x2 = x2.float()
                x3 = x3.float()
                x4 = x4.float()
                x5 = x5.float()
                inputs_batchsize = [x1,x2,x3,x4,x5]
                z_x1, z_x2, z_x3, z_x4, z_x5, mu1, mu2, mu3, mu4, mu5, \
                logvar1, logvar2, logvar3, logvar4, logvar5, z_s = encoder(inputs_batchsize)

                recon_x1, recon_x2, recon_x3, recon_x4, recon_x5, \
                recon_s1, recon_s2, recon_s3, recon_s4, recon_s5 = decoder(z_x1,z_x2,z_x3,z_x4,z_x5,z_s)

                # loss
                loss = loss_function_N(x1, x2, x3, x4, x5, recon_x1, recon_x2, recon_x3, recon_x4, recon_x5,
                                     mu1, mu2, mu3, mu4, mu5, logvar1, logvar2, logvar3, logvar4, logvar5,
                                     beta, delta, z_s, alpha, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                all_loss = train_loss / (batch_idx + 1)
                trainbar.set_description('l:%.3f' % (all_loss))
                sleep(0.05)


    if data_name == 'Caltech101':
        encoder = encoder_cada_C(input_dim=input_dim, hidden_dim=hidden_dim, z=z).to(device)
        decoder = decoder_cada_C(input_dim=input_dim, hidden_dim=hidden_dim, z=z).to(device)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.00015, amsgrad=True)
        for i in range(epoch):
            print('epoch',i)
            trainbar = tqdm(trainloader)
            encoder.train()
            decoder.train()
            train_loss = 0
            if i > 5 and i < 21:
                delta += 0.54  # da_loss
            if i < 91:
                beta += 0.0026

            for batch_idx, (x1, y, x2, x3, x4, x5, x6) in enumerate(trainbar):
                x1 = x1.float()  # print(x.size())  (64,3180)
                x2 = x2.float()
                x3 = x3.float()
                x4 = x4.float()
                x5 = x5.float()
                x6 = x6.float()
                inputs_batchsize = [x1, x2, x3, x4, x5, x6]
                z_x1, z_x2, z_x3, z_x4, z_x5, z_x6, mu1, mu2, mu3, mu4, mu5, mu6, \
                logvar1, logvar2, logvar3, logvar4, logvar5, logvar6, z_s = encoder(inputs_batchsize)

                recon_x1, recon_x2, recon_x3, recon_x4, recon_x5, recon_x6, \
                recon_s1, recon_s2, recon_s3, recon_s4, recon_s5, recon_s6 = decoder(z_x1, z_x2, z_x3, z_x4, z_x5,
                                                                                       z_x6, z_s)
                # loss
                loss = loss_function_C(x1, x2, x3, x4, x5, x6, recon_x1, recon_x2, recon_x3, recon_x4, recon_x5,recon_x6,
                                     mu1, mu2, mu3, mu4, mu5, mu6, logvar1, logvar2, logvar3, logvar4, logvar5, logvar6,
                                     beta, delta, z_s, alpha, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5,recon_s6)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                all_loss = train_loss / (batch_idx + 1)
                trainbar.set_description('l:%.3f' % (all_loss))
                sleep(0.05)


    if data_name == 'ALOI':
        encoder = encoder_cada_A(input_dim=input_dim, hidden_dim=hidden_dim, z=z).to(device)
        decoder = decoder_cada_A(input_dim=input_dim, hidden_dim=hidden_dim, z=z).to(device)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.00015, amsgrad=True)
        for i in range(epoch):
            print('epoch',i)
            trainbar = tqdm(trainloader)
            encoder.train()
            decoder.train()
            train_loss = 0
            if i > 5 and i < 21:
                delta += 0.54  # da_loss
            if i < 91:
                beta += 0.0026

            for batch_idx, (x1, y, x2, x3, x4) in enumerate(trainbar):
                x1 = x1.float()  # print(x.size())  (64,3180)
                x2 = x2.float()
                x3 = x3.float()
                x4 = x4.float()
                inputs_batchsize = [x1, x2, x3, x4]
                z_x1, z_x2, z_x3, z_x4, mu1, mu2, mu3, mu4, \
                logvar1, logvar2, logvar3, logvar4, z_s = encoder(inputs_batchsize)

                recon_x1, recon_x2, recon_x3, recon_x4, \
                recon_s1, recon_s2, recon_s3, recon_s4 = decoder(z_x1, z_x2, z_x3, z_x4, z_s)
                # loss
                loss = loss_function_A(x1, x2, x3, x4, recon_x1, recon_x2, recon_x3, recon_x4,
                                     mu1, mu2, mu3, mu4, logvar1, logvar2, logvar3, logvar4,
                                     beta, delta, z_s, alpha, recon_s1, recon_s2, recon_s3, recon_s4)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                all_loss = train_loss / (batch_idx + 1)
                trainbar.set_description('l:%.3f' % (all_loss))
                sleep(0.05)


# ================================================================================================
def train_fune(dataname,epoch,load_path,trainloader,inputs, Y):
    list_acc = []
    list_nmi = []
    list_ari = []
    list_pur = []
    Y1 = np.reshape(Y, [Y.shape[0]])

    if dataname == 'MSRCv1':
        pthfile = load_path + '/eMSRCv1.pth'
        pthfile1 = load_path + '/dMSRCv1.pth'
        encoder_1 = torch.load(pthfile)
        decoder_1 = torch.load(pthfile1)
        model_cluster1 = joblib.load(load_path + 'MSRCv1.pkl')

        cluster_centers = Variable((torch.from_numpy(model_cluster1.cluster_centers_).type(torch.FloatTensor)).cuda(),requires_grad=True)
        optimizer1 = optim.SGD(list(encoder_1.parameters()) + list(decoder_1.parameters()) + [cluster_centers], lr=1e-3)

        for i in range(epoch):
            print("epoch:", i)
            trainbar = tqdm(trainloader)
            encoder_1.train()
            decoder_1.train()
            train_loss = 0
            for batch_idx, (x1, y, x2, x3, x4, x5) in enumerate(trainbar):
                x1 = x1.float()  # print(x.size())  (64,3180)
                x2 = x2.float()
                x3 = x3.float()
                x4 = x4.float()
                x5 = x5.float()
                inputs_batchsize = [x1, x2, x3, x4, x5]

                z_x1, z_x2, z_x3, z_x4, z_x5, mu1, mu2, mu3, mu4, mu5, \
                logvar1, logvar2, logvar3, logvar4, logvar5, z_s = encoder_1(inputs_batchsize)

                recon_x1, recon_x2, recon_x3, recon_x4, recon_x5, \
                recon_s1, recon_s2, recon_s3, recon_s4, recon_s5 = decoder_1(z_x1,z_x2,z_x3,z_x4,z_x5,z_s)

                zs_loss = ZS_loss_M(x1, x2, x3, x4, x5, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5)

                loss2, _ = loss_func(z_s,cluster_centers)
                loss = loss2 * 10 + zs_loss * 0.01

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                train_loss += loss.item()
                all_loss = train_loss / (batch_idx + 1)
                trainbar.set_description('l:%.3f' % (all_loss))
                sleep(0.05)

            #  ================clustering===================
            _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, z_s1 = encoder_1(inputs)
            _, p = loss_func(z_s1,cluster_centers)
            pred_label = dist_2_label(p)
            accuracy = metrics.acc(Y, pred_label)
            nmi1 = metrics.nmi(Y1, pred_label)
            purity1 = purity_score(Y1, pred_label)
            ari1 = metrics.ari(Y1, pred_label)

            list_acc.append(accuracy)
            list_nmi.append(nmi1)
            list_ari.append(ari1)
            list_pur.append(purity1)
            

    if dataname == 'Yale':
        pthfile = load_path + '/eYale.pth'
        pthfile1 = load_path + '/dYale.pth'
        encoder_1 = torch.load(pthfile)
        decoder_1 = torch.load(pthfile1)
        model_cluster1 = joblib.load(load_path + 'Yale.pkl')

        cluster_centers = Variable((torch.from_numpy(model_cluster1.cluster_centers_).type(torch.FloatTensor)).cuda(),requires_grad=True)
        optimizer1 = optim.SGD(list(encoder_1.parameters()) + list(decoder_1.parameters()) + [cluster_centers], lr=1e-4)  # 1e-4

        for i in range(epoch):
            print("epoch:", i)
            trainbar = tqdm(trainloader)
            encoder_1.train()
            decoder_1.train()
            train_loss = 0
            for batch_idx, (x1, y, x2, x3) in enumerate(trainbar):
                x1 = x1.float()  # print(x.size())  (64,3180)
                x2 = x2.float()
                x3 = x3.float()
                inputs_batchsize = [x1,x2,x3]
                z_x1, z_x2, z_x3, mu1, mu2, mu3, \
                logvar1, logvar2, logvar3, z_s = encoder_1(inputs_batchsize)

                recon_x1, recon_x2, recon_x3, \
                recon_s1, recon_s2, recon_s3 = decoder_1(z_x1, z_x2, z_x3, z_s)

                zs_loss = ZS_loss_Y(x1, x2, x3, recon_s1, recon_s2, recon_s3)

                loss2, _ = loss_func(z_s,cluster_centers)
                loss = loss2 * 10 + zs_loss * 0.1  # 10    0.1

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                train_loss += loss.item()
                all_loss = train_loss / (batch_idx + 1)
                trainbar.set_description('l:%.3f' % (all_loss))
                sleep(0.05)

            #  ================clustering===================
            encoder_1.eval()
            decoder_1.eval()
            with torch.no_grad():
                _, _, _, _, _, _, _, _, _, z_s1 = encoder_1(inputs)
            _, p = loss_func(z_s1,cluster_centers)
            pred_label = dist_2_label(p)
            accuracy = metrics.acc(Y, pred_label)
            nmi1 = metrics.nmi(Y1, pred_label)
            purity1 = purity_score(Y1, pred_label)
            ari1 = metrics.ari(Y1, pred_label)

            list_acc.append(accuracy)
            list_nmi.append(nmi1)
            list_ari.append(ari1)
            list_pur.append(purity1)
            

    if dataname == 'NUS':
        pthfile = load_path + '/eNUS.pth'
        pthfile1 = load_path + '/dNUS.pth'
        encoder_1 = torch.load(pthfile)
        decoder_1 = torch.load(pthfile1)
        model_cluster1 = joblib.load(load_path + 'NUS.pkl')

        cluster_centers = Variable((torch.from_numpy(model_cluster1.cluster_centers_).type(torch.FloatTensor)).cuda(),requires_grad=True)
        optimizer1 = optim.SGD(list(encoder_1.parameters()) + list(decoder_1.parameters()) + [cluster_centers], lr=1e-5)

        for i in range(epoch):
            print("epoch:", i)
            trainbar = tqdm(trainloader)
            encoder_1.train()
            decoder_1.train()
            train_loss = 0
            for batch_idx, (x1, y, x2, x3, x4, x5) in enumerate(trainbar):
                x1 = x1.float()  # print(x.size())  (64,3180)
                x2 = x2.float()
                x3 = x3.float()
                x4 = x4.float()
                x5 = x5.float()
                inputs_batchsize = [x1, x2, x3,x4,x5]
                z_x1, z_x2, z_x3, z_x4, z_x5, mu1, mu2, mu3, mu4, mu5, \
                logvar1, logvar2, logvar3, logvar4, logvar5, z_s = encoder_1(inputs_batchsize)

                recon_x1, recon_x2, recon_x3, recon_x4, recon_x5, \
                recon_s1, recon_s2, recon_s3, recon_s4, recon_s5 = decoder_1(z_x1, z_x2, z_x3, z_x4, z_x5, z_s)

                zs_loss = ZS_loss_N(x1, x2, x3, x4, x5, z_s, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5)

                loss2, _ = loss_func(z_s,cluster_centers)
                loss = loss2 * 10 + zs_loss * 0.01

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                train_loss += loss.item()
                all_loss = train_loss / (batch_idx + 1)
                trainbar.set_description('l:%.3f' % (all_loss))
                sleep(0.05)
            #  ================clustering===================
            encoder_1.eval()
            decoder_1.eval()
            with torch.no_grad():
                _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, z_s1 = encoder_1(inputs)
            # z_s = z_s1.cpu().detach().numpy()
            _, p = loss_func(z_s1,cluster_centers)
            pred_label = dist_2_label(p)
            accuracy = metrics.acc(Y, pred_label)
            nmi1 = metrics.nmi(Y1, pred_label)
            purity1 = purity_score(Y1, pred_label)
            ari1 = metrics.ari(Y1, pred_label)

            list_acc.append(accuracy)
            list_nmi.append(nmi1)
            list_ari.append(ari1)
            list_pur.append(purity1)
            

    if dataname == 'Caltech101':
        pthfile = load_path + '/eCaltech101.pth'
        pthfile1 = load_path + '/dCaltech101.pth'
        encoder_1 = torch.load(pthfile)
        decoder_1 = torch.load(pthfile1)
        model_cluster1 = joblib.load(load_path + 'Caltech101.pkl')

        cluster_centers = Variable((torch.from_numpy(model_cluster1.cluster_centers_).type(torch.FloatTensor)).cuda(),requires_grad=True)
        optimizer1 = optim.SGD(list(encoder_1.parameters()) + list(decoder_1.parameters()) + [cluster_centers], lr=1e-4)

        for i in range(epoch):
            print("epoch:", i)
            trainbar = tqdm(trainloader)
            encoder_1.train()
            decoder_1.train()
            train_loss = 0
            for batch_idx, (x1, y, x2, x3, x4, x5, x6) in enumerate(trainbar):
                x1 = x1.float()  # print(x.size())  (64,3180)
                x2 = x2.float()
                x3 = x3.float()
                x4 = x4.float()
                x5 = x5.float()
                x6 = x6.float()
                inputs_batchsize = [x1, x2, x3, x4, x5,x6]
                z_x1, z_x2, z_x3, z_x4, z_x5, z_x6, mu1, mu2, mu3, mu4, mu5, mu6, \
                logvar1, logvar2, logvar3, logvar4, logvar5, logvar6, z_s = encoder_1(inputs_batchsize)

                recon_x1, recon_x2, recon_x3, recon_x4, recon_x5, recon_x6, \
                recon_s1, recon_s2, recon_s3, recon_s4, recon_s5, recon_s6 = decoder_1(z_x1, z_x2, z_x3, z_x4, z_x5,
                                                                                       z_x6, z_s)
                zs_loss = ZS_loss_C(x1, x2, x3, x4, x5, x6, z_s, recon_s1, recon_s2, recon_s3, recon_s4, recon_s5,
                                  recon_s6)
                loss2, _ = loss_func(z_s,cluster_centers)
                loss = loss2 * 100 + zs_loss * 0.01

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                train_loss += loss.item()
                all_loss = train_loss / (batch_idx + 1)
                trainbar.set_description('l:%.3f' % (all_loss))
                sleep(0.05)
            #  ================clustering===================
            encoder_1.eval()
            decoder_1.eval()
            with torch.no_grad():
                _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, z_s1 = encoder_1(inputs)
            _, p = loss_func(z_s1,cluster_centers)
            pred_label = dist_2_label(p)
            accuracy = metrics.acc(Y, pred_label)
            nmi1 = metrics.nmi(Y1, pred_label)
            purity1 = purity_score(Y1, pred_label)
            ari1 = metrics.ari(Y1, pred_label)

            list_acc.append(accuracy)
            list_nmi.append(nmi1)
            list_ari.append(ari1)
            list_pur.append(purity1)
            

    if dataname == 'ALOI':
        pthfile = load_path + '/eALOI.pth'
        pthfile1 = load_path + '/dALOI.pth'
        encoder_1 = torch.load(pthfile)
        decoder_1 = torch.load(pthfile1)
        model_cluster1 = joblib.load(load_path + 'ALOI.pkl')

        cluster_centers = Variable((torch.from_numpy(model_cluster1.cluster_centers_).type(torch.FloatTensor)).cuda(),requires_grad=True)
        optimizer1 = optim.SGD(list(encoder_1.parameters()) + list(decoder_1.parameters()) + [cluster_centers], lr=1e-4)

        for i in range(epoch):
            print("epoch:", i)
            trainbar = tqdm(trainloader)
            encoder_1.train()
            decoder_1.train()
            train_loss = 0
            for batch_idx, (x1, y, x2, x3, x4) in enumerate(trainbar):
                x1 = x1.float()  # print(x.size())  (64,3180)
                x2 = x2.float()
                x3 = x3.float()
                x4 = x4.float()
                inputs_batchsize = [x1, x2, x3, x4]
                z_x1, z_x2, z_x3, z_x4, mu1, mu2, mu3, mu4, \
                logvar1, logvar2, logvar3, logvar4, z_s = encoder_1(inputs_batchsize)

                recon_x1, recon_x2, recon_x3, recon_x4, \
                recon_s1, recon_s2, recon_s3, recon_s4 = decoder_1(z_x1, z_x2, z_x3, z_x4, z_s)

                zs_loss = ZS_loss_A(x1, x2, x3, x4, z_s, recon_s1, recon_s2, recon_s3, recon_s4)

                loss2, _ = loss_func(z_s,cluster_centers)
                loss = loss2 * 10 + zs_loss * 0.01

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                train_loss += loss.item()
                all_loss = train_loss / (batch_idx + 1)
                trainbar.set_description('l:%.3f' % (all_loss))
                sleep(0.05)
            #  ================clustering===================
            encoder_1.eval()
            decoder_1.eval()
            with torch.no_grad():
                _, _, _, _, _, _, _, _, _, _, _, _, z_s1 = encoder_1(inputs)
            _, p = loss_func(z_s1,cluster_centers)
            pred_label = dist_2_label(p)
            accuracy = metrics.acc(Y, pred_label)
            nmi1 = metrics.nmi(Y1, pred_label)
            purity1 = purity_score(Y1, pred_label)
            ari1 = metrics.ari(Y1, pred_label)

            list_acc.append(accuracy)
            list_nmi.append(nmi1)
            list_ari.append(ari1)
            list_pur.append(purity1)
            

    mean_acc = np.mean(list_acc)
    mean_nmi = np.mean(list_nmi)
    mean_ari = np.mean(list_ari)
    mean_pur = np.mean(list_pur)
    print('\nMEAN_ACC:', mean_acc)
    print('MEAN_NMI:', mean_nmi)
    print('MEAN_ARI:', mean_ari)
    print('MEAN_purity:', mean_pur)

    var_acc = np.var(list_acc)
    var_nmi = np.var(list_nmi)
    var_ari = np.var(list_ari)
    var_pur = np.var(list_pur)
    print('\nVAR_ACC:', var_acc * 100)
    print('VAR_NMI:', var_nmi * 100)
    print('VAR_ARI:', var_ari * 100)
    print('VAR_purity:', var_pur * 100)

    std_acc = np.std(list_acc)
    std_nmi = np.std(list_nmi)
    std_ari = np.std(list_ari)
    std_pur = np.std(list_pur)
    print('\nSTD_ACC:', std_acc * 100)
    print('STD_NMI:', std_nmi * 100)
    print('STD_ARI:', std_ari * 100)
    print('STD_purity:', std_pur * 100)