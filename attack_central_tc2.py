import torch
# torch.nn lib
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# torch.optim lib
import torch.optim as optim
# torch.utils lib
from torch.utils.data import DataLoader
# torchvision lib
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from torchvision import datasets, transforms

# torchplus lib
from torchplus.utils import save_image2
from torchplus.nn import PixelLoss

# other lib
from tqdm import tqdm
from piq import SSIMLoss
import os
import copy
import time
import warnings
import random

# Tắt các cảnh báo FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# args
from utils.options import args_parser
# Nets
from models.Nets import Inversion, PowerAmplification, CNNMnist

if __name__ == "__main__":

    args = args_parser()
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")

    if args.dataset == "mnist":
        # trans_mnist = transforms.Compose([transforms.ToTensor()])
        trans_mnist = Compose([Grayscale(num_output_channels=1), Resize((32, 32)), ToTensor()])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == "fashion-mnist":
        # trans_fashion_mnist = transforms.Compose([transforms.ToTensor()])
        trans_fashion_mnist = Compose([Grayscale(num_output_channels=1), Resize((32, 32)), ToTensor()])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                             transform=trans_fashion_mnist)
    elif args.dataset == "cifar":
        pass
    else:
        exit('Error: unrecognized dataset')

    print(len(dataset_train))
    print(len(dataset_test))

    train_dl = DataLoader(
        dataset=dataset_train,
        batch_size=args.bs,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True,
    )

    test_dl = DataLoader(
        dataset=dataset_test,
        batch_size=args.bs,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    # attack Central
    if args.dataset == "mnist":
        print("Attack central ds mnist...")
        # target_pkl = "./central_model/central_model_mnist_epoch10.pkl"  # central
        # fed global 100
        target_pkl = "/root/Project/global_model_100.pkl"
    elif args.dataset == "fashion-mnist":
        print("Attack central ds fashion-mnist...")
        target_pkl = "./central_model/central_model_fashion-mnist_epoch100.pkl"
    elif args.dataset == "cifar":
        pass
    # alpha
    alpha = [1.0, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 1500, 2000]
    # create diction saving evaluation
    ssimloss_li = []
    mseloss_li = []
    pixelloss_li = []
    for id_alpha in alpha:
        # classifier
        target_classifier = CNNMnist(args).train(False).to(args.device)
        # amplification
        target_amplification = (
            PowerAmplification(args.num_classes, 1 / id_alpha).train(False).to(args.device)
        )
        target_classifier.load_state_dict(torch.load(open(target_pkl, "rb"), map_location=args.device))
        target_classifier.requires_grad_(False)
        # model attack
        myinversion = copy.deepcopy(Inversion(args).train(True)).to(args.device)
        # optimizer
        optimizer = optim.Adam(
            myinversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
        )
        print(f"Alpha = 1/{id_alpha}")
        # training 100 epochs
        for epoch_id in tqdm(range(1, args.epochs + 1), desc="Total Epoch"):
            total_loss = 0
            t_start = time.time()
            for i, (im, label) in enumerate(tqdm(test_dl, desc=f"epoch {epoch_id}")):
                im = im.to(args.device)
                label = label.to(args.device)
                optimizer.zero_grad()
                out = target_classifier.forward(im)
                after_softmax = F.softmax(out, dim=-1)
                after_softmax = target_amplification.forward(after_softmax)
                rim = myinversion.forward(after_softmax)
                loss = F.mse_loss(rim, im)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            t_end = time.time()
            # time for each epoch
            during_time = t_end - t_start
            print(f"Epoch {epoch_id}/{args.epochs}, Loss: {total_loss / len(test_dl)}, Time: {during_time}")
        # evaluation after training 100 epochs
        with torch.no_grad():
            myinversion.eval()
            ssim_eval = 0
            mse_eval = 0
            pixel_eval = 0
            for i, (im, label) in enumerate(tqdm(train_dl, desc=f"epoch {epoch_id}")):
                im = im.to(args.device)
                label = label.to(args.device)
                out = target_classifier.forward(im)
                after_softmax = F.softmax(out, dim=-1)
                after_softmax = target_amplification.forward(after_softmax)
                rim = myinversion.forward(after_softmax)
                # save images
                if i == 0:
                    images_path = f"./attack_tc2_label4_fed/{args.dataset}/save_images"
                    os.makedirs(images_path, exist_ok=True)
                    tensor_id = torch.where(label == 4)[0]
                    save_image2(rim[tensor_id].detach(), f"{images_path}/output/{id_alpha}/image_epoch_class4.png")
                # evaluation in train_dl
                ssim_eval += SSIMLoss()(rim, im).item()
                mse_eval += F.mse_loss(rim, im).item()
                pixel_eval += PixelLoss(13)(rim, im).item()
            # evaluation for each alpha
            ssim = ssim_eval / len(train_dl)
            mse = mse_eval / len(train_dl)
            pixel = pixel_eval / len(train_dl)
            print(f"Alpha: 1/{id_alpha}, ssim: {ssim}, mse: {mse}, pixel: {pixel}")
        ssimloss_li.append(round(ssim, 4))
        mseloss_li.append(round(mse , 4))
        pixelloss_li.append(round(pixel, 4))
    with open(os.path.join(f"./attack_tc2_label4_fed/{args.dataset}", "evaluation.pkl"), "wb") as f:
        torch.save((ssimloss_li, mseloss_li, pixelloss_li), f)



















