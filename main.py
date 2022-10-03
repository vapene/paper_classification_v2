import argparse
import numpy as np
import torch
import os
import torch.nn.functional as F
import torch.optim as optim
import scipy
from torchvision import datasets, transforms
from PIL import Image
from param_parser import parameter_parser, tab_printer
from utils import RandomRotation, PaperDataset, EMA, to_one_hot
from models import M3, M5, M7, resnet , alex, vgg, dense
from tensorboardX import SummaryWriter
writer = SummaryWriter()

def train(args, score_dict, prediction_dict, target_list, trial, MODEL, SEED, EPOCHS):
    ### create file & path
    if not os.path.exists(f"./outcome2"):
        os.makedirs(f"./outcome2")
    MODEL_FILE = str(f"./outcome2/{MODEL}_{trial}.pt")

    ### transform
    transform = transforms.Compose([
        RandomRotation(20, seed=SEED),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5)])
    ### loader
    train_dataset = PaperDataset(training=True, transform=transform)
    test_dataset = PaperDataset(training=False, transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)  # 800:6,
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    ### hyper-parameter
    model = eval(f"{MODEL}().to(device)")
    ema = EMA(model, decay=0.999)  # Exponential Weighted Moving Average
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    g_step = 0
    ### training

    best_pred_matrix = torch.zeros(len(test_loader.dataset),1)
    max_correct = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0


        for batch_idx, (data,target,idx) in enumerate(train_loader):
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output,target)
            train_pred = output.argmax(dim = 1, keepdim = True)
            train_correct += train_pred.eq(target.view_as(train_pred)).sum().item()
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            loss.backward()
            optimizer.step()
            g_step +=1
            ema(model, g_step)

        train_accuracy = 100 * train_correct / (len(train_loader.dataset)-(len(train_loader.dataset)%args.batch_size))

        ### test
        model.eval()
        ema.assign(model)
        test_loss = 0
        test_correct = 0
        pred_matrix = torch.zeros(len(test_loader.dataset)-(len(test_loader.dataset)%args.batch_size) ,1)
        start = 0
        end = 0
        with torch.no_grad():
            for batch_idx, (data, target, idx) in enumerate(test_loader):
                data, target = data.to(device), target.to(device, dtype=torch.int64)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                ##
                start = batch_idx * args.batch_size
                end = start + args.batch_size
                pred_matrix[start:end] = pred

                ##
                if len(target_list) < end:
                    target_list.extend(target.detach().cpu())
                ##
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        ema.resume(model)
        if (max_correct < test_correct):
            max_correct = test_correct
            print(f"Best accuracy! correct images: {test_correct}")
            best_pred_matrix = pred_matrix
        test_accuracy = 100 * test_correct / end
        writer.add_scalar(f"{MODEL}/loss/test_acc", test_accuracy, epoch)

        best_test_accuracy = 100 * max_correct / end
        lr_scheduler.step()


    best_pred_matrix = best_pred_matrix.to(torch.int64)
    one_hot = to_one_hot(best_pred_matrix.flatten(), 5)
    try:
        score_dict[f"{MODEL}"].extend([best_test_accuracy])
        prediction_dict[f"{MODEL}"].extend([one_hot])
        # target_dict[f"{MODEL}"].extend([target_list])
    except:
        score_dict.update({f"{MODEL}": [best_test_accuracy]})
        prediction_dict.update({f"{MODEL}": [one_hot]})
        # target_dict.update({f"{MODEL}": [target_list]})

    # score_dict[MODEL] = score_dict[MODEL]+torch.Tensor([best_test_accuracy])
    return score_dict, prediction_dict, target_list



def ensemble(args, SCORE_DICT, PREDICTION_DICT, TARGET_LIST):
    homo_pred_dict = {}
    homo_score_dict = {}
    hetero_pred = 0
    for model in args.models:
        homo_pred_dict[model] = 0
        homo_score_dict[model] = 0

    target_list = torch.Tensor([t.item() for t in TARGET_LIST])
    for model in args.models:
        for index in range(len(SCORE_DICT[model])):
            homo_pred_dict[model] += np.array(PREDICTION_DICT[model][index]) * SCORE_DICT[model][index]
            homo_score_dict[model] += SCORE_DICT[model][index]
        homo_pred_dict[model] = to_one_hot(torch.Tensor(np.argmax(homo_pred_dict[model], axis=1)),len(PREDICTION_DICT[model][index][0]))
        hetero_pred += np.array(homo_pred_dict[model])*homo_score_dict[model]
    final_pred = torch.IntTensor(np.argmax(hetero_pred, axis=1))
    final_correct = final_pred.eq(target_list.view_as(final_pred)).sum().item()
    final_accuracy = final_correct / len(TARGET_LIST)

    return final_accuracy


if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + args.device)
    # score_dict = {"M3":torch.Tensor([1e-10]), "M5":torch.Tensor([1e-10]), "M7":torch.Tensor([1e-10]), "resnet":torch.Tensor([1e-10])}
    score_dict = {}
    prediction_dict= {}

    # prediction_dict = {"M3": torch.zeros((1000, 5)), "M5": torch.zeros((1000, 5)), "M7": torch.zeros((1000, 5)), "resnet": torch.zeros((1000, 5))}
    target_list = []
    for trial in range(args.trial): # 3
        for model in args.models:
            score_dict, prediction_dict, target_list = train(args, score_dict, prediction_dict, target_list, trial, model, args.seed, args.epochs)
        print('trial:',trial,"model:",model)
    final_accuracy = ensemble(args, score_dict, prediction_dict, target_list)
    print('score_dict',score_dict,'\n final_accuracy',final_accuracy)
    writer.close()