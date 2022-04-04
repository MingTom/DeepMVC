import os
from model_base.train import *
from dataloader import *
import argparse
import warnings

warnings.filterwarnings("ignore")

load_path = './loadpth/'
DWAMVC = "DWAMVC:Enter the dataset name"
parser = argparse.ArgumentParser(description=DWAMVC)
parser.add_argument("--data_name", default="MSRCv1", help="[MSRCv1, Yale, NUS, ALOI, Caltech101]", type=str)
parser.add_argument("--gpu", default='0', help="Specify GPU", type=str)
parser.add_argument("--epochs_t", default=500, help="Training epoch", type=int)
parser.add_argument("--epochs_f", default=10, help="Fine-tuning epoch", type=int)
parser.add_argument("--train", default=False, help="train[True] or load_finetune[False]")
args = parser.parse_args()

epochs = args.epochs_t
epochs1 = args.epochs_f
choose_gpu = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = choose_gpu


if args.data_name == 'MSRCv1':
    from model_base.M_loadmodel import *
    _, _, _, _, _, X1, X2, X3, X4, X5, Y = dataset_read(args.data_name)
    data_set = DL_MSRCv1(args.data_name)
    classes = 7
    z = 7
    alpha = 10
    hidden_dim = [500, 500, 1024]
    input_dim = data_set.__getlen__()
    batch_size = 32
    inputs = [X1, X2, X3, X4, X5]
    trainloader = Data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    if args.train is False:
        train_fune(args.data_name, epochs1, load_path, trainloader, inputs, Y)
    else:
        train(epochs, input_dim, hidden_dim, z, trainloader,inputs, Y, classes=classes, data_name=args.data_name, alpha=alpha)

if args.data_name == 'Yale':
    from model_base.Y_loadmodel import *
    _, _, _, X1, X2, X3, Y = dataset_read(args.data_name)
    data_set = DL_Yale(args.data_name)
    input_dim = data_set.__getlen__()
    classes = 15
    z = 15
    alpha = 10
    hidden_dim = [2000, 1024, 500]
    batch_size = 16
    inputs = [X1, X2, X3]
    trainloader = Data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    if args.train is False:
        train_fune(args.data_name,epochs1,load_path,trainloader,inputs,Y)
    else:
        train(epochs, input_dim, hidden_dim, z, trainloader, inputs, Y, classes=classes, data_name=args.data_name,alpha=alpha)

if args.data_name == 'NUS':
    from model_base.N_loadmodel import *
    _, _, _, _, _, X1, X2, X3, X4, X5, Y = dataset_read(args.data_name)
    data_set = DL_NUS(args.data_name)
    input_dim = data_set.__getlen__()
    classes = 31
    z = 31
    alpha = 10
    hidden_dim = [500, 500, 1024]
    batch_size = 64
    inputs = [X1, X2, X3, X4, X5]
    trainloader = Data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    if args.train is False:
        train_fune(args.data_name, epochs1, load_path, trainloader, inputs, Y)
    else:
        train(epochs, input_dim, hidden_dim, z, trainloader, inputs, Y, classes=classes, data_name=args.data_name,alpha=alpha)

if args.data_name == 'Caltech101':
    from model_base.C_loadmodel import *
    _, _, _, _, _, _, X1, X2, X3, X4, X5, X6, Y = dataset_read(args.data_name)
    data_set = DL_Caltech101(args.data_name)
    input_dim = data_set.__getlen__()
    classes = 102
    z = 10
    alpha = 20
    hidden_dim = [500, 500, 1024]
    batch_size = 64
    inputs = [X1, X2, X3, X4, X5, X6]
    trainloader = Data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    if args.train is False:
        train_fune(args.data_name, epochs1, load_path, trainloader, inputs, Y)
    else:
        train(epochs, input_dim, hidden_dim, z, trainloader, inputs, Y, classes=classes, data_name=args.data_name,alpha=alpha)

if args.data_name == 'ALOI':
    from model_base.A_loadmodel import *
    _, _, _, _, X1, X2, X3, X4, Y = dataset_read(args.data_name)
    data_set = DL_ALOI(args.data_name)
    input_dim = data_set.__getlen__()
    classes = 100
    z = 10
    alpha = 20
    hidden_dim = [500, 500, 1024]
    batch_size = 200
    inputs = [X1, X2, X3, X4]
    trainloader = Data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    if args.train is False:
        train_fune(args.data_name, epochs1, load_path, trainloader, inputs, Y)
    else:
        train(epochs, input_dim, hidden_dim, z, trainloader, inputs, Y, classes=classes, data_name=args.data_name,alpha=alpha)




