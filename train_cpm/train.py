import sys
import os
sys.path.append(os.getcwd())
print(sys.path)
from tqdm import tqdm
from numpy import mod
from pickle import NONE
import torch.nn as nn
import torch.utils.data.dataloader
from train_cpm.utils import AverageMeter
from preprocess.Transformers import Compose, RandomCrop, RandomResized, TestResized
from preprocess.gen_data import LSP_DATA
import copy
from cpm import cpm, cpm_condense



def train_model(training_dataset_path, val_data_path, model, criterion, optimizer, device=None, ts_mode=False, model_teacher=None, max_epoch=300, eps=1e-8, batch_size=1, save_name='default'):
    if ts_mode is True:
        save_name = save_name+'_ts'
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_dataset_path = training_dataset_path
    val_data_path = val_data_path
    model_save_path = os.path.join(os.getcwd(), 'model\\'+save_name+'.pth')
    best_model_path = os.path.join(
        os.getcwd(), 'model\\best_'+save_name+'.pth')

    criterion = criterion.to(device)

    model = model.to(device)

    optimizer = optimizer

    train_losses = AverageMeter()
    val_losses = AverageMeter()
    min_losses = 999.0

    epoch = 0
    try:
        model = torch.load(best_model_path).to(device)
        print("continue trainning from the last checkpoint ...")
    except:
        pass

    val_loss_last = 999.0
    train_loss_last = 999.0
    data_train = LSP_DATA('lsp', training_dataset_path, 8,
                          Compose([RandomResized(), RandomCrop(368), TestResized(368)]))
    train_loader = torch.utils.data.dataloader.DataLoader(
        data_train, batch_size=batch_size)
    data_val = LSP_DATA('lsp', val_data_path, 8, Compose([TestResized(368)]))
    val_loader = torch.utils.data.dataloader.DataLoader(
        data_val, batch_size=batch_size)

    while epoch < max_epoch:
        print('epoch ', epoch)
        """--------Train--------"""
        # Training data

        for j, data in tqdm(enumerate(train_loader)):
            inputs, heatmap, centermap, _ = data

            inputs = inputs.to(device)
            heatmap = heatmap.to(device)
            centermap = centermap.to(device)

            input_var = torch.autograd.Variable(inputs)
            heatmap_var = torch.autograd.Variable(heatmap)
            centermap_var = torch.autograd.Variable(centermap)

            heat1, heat2, heat3, heat4, heat5, heat6 = model(
                input_var, centermap_var)
            if ts_mode is True:
                with torch.no_grad():
                    heat11, heat22, heat33, heat44, heat55, heat66 = model_teacher(
                        input_var, centermap_var)
                loss1 = criterion(heat1, heat11)
                loss2 = criterion(heat2, heat22)
                loss3 = criterion(heat3, heat33)
                loss4 = criterion(heat4, heat44)
                loss5 = criterion(heat5, heat55)
                loss6 = criterion(heat6, heat66)

            else:
                loss1 = criterion(heat1, heatmap_var)
                loss2 = criterion(heat2, heatmap_var)
                loss3 = criterion(heat3, heatmap_var)
                loss4 = criterion(heat4, heatmap_var)
                loss5 = criterion(heat5, heatmap_var)
                loss6 = criterion(heat6, heatmap_var)

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            train_losses.update(loss.item(), inputs.size(0))
            if j % 100 == 0:
                print('Train Loss: ', train_losses.avg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Loss: ', train_losses.avg)
        torch.save(model, model_save_path)

        # --------Validation--------
        # Validation
        print('-----------Validation-----------')
        # Validation data
        model.eval()
        for j, data in enumerate(val_loader):
            inputs, heatmap, centermap, _ = data

            inputs = inputs.to(device)
            heatmap = heatmap.to(device)
            centermap = centermap.to(device)

            input_var = torch.autograd.Variable(inputs)
            heatmap_var = torch.autograd.Variable(heatmap)
            centermap_var = torch.autograd.Variable(centermap)

            heat1, heat2, heat3, heat4, heat5, heat6 = model(
                input_var, centermap_var)

            loss1 = criterion(heat1, heatmap_var)
            loss2 = criterion(heat2, heatmap_var)
            loss3 = criterion(heat3, heatmap_var)
            loss4 = criterion(heat4, heatmap_var)
            loss5 = criterion(heat5, heatmap_var)
            loss6 = criterion(heat6, heatmap_var)

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            val_losses.update(loss.item(), inputs.size(0))

        print('Validation Loss: ', val_losses.avg)
        if val_losses.avg < min_losses:
            # Save best cpm
            torch.save(model, best_model_path)
            min_losses = val_losses.avg
        if abs(val_losses.avg-val_loss_last) < eps and abs(train_losses.avg-train_loss_last) < eps:
            break
        val_loss_last = val_losses.avg
        train_loss_last = train_losses.avg
        model.train()

        epoch += 1


def transfer_state_dict(pretrained_dict, model_dict):
    '''
    According to model_dict, remove some unneeded parameters of pretrained_dict in order to migrate to the new network
    '''
    state_dict = {}
    for k, v in pretrained_dict.state_dict().items():
        if k in model_dict.keys() and pretrained_dict.state_dict()[k].shape == model_dict[k].shape:
            state_dict[k] = v
        else:
            print("Missing key(s) or dismatch shape in state_dict :{}".format(k))
    return state_dict


def transfer_model(pretrained_file, model):
    '''
    Import only parameters of the same name and shape in pretrained_model
    '''
    pretrained_dict = torch.load(pretrained_file)  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # Before update, you need to remove some unneeded parameters of pretrained_dict
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # update parameter
    model.load_state_dict(model_dict)
    return model


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


if __name__ == "__main__":
    # set dataset path
    training_dataset_path = 'atrw_split\\trainset'
    val_data_path = 'atrw_split\\valset'


    # set criterion
    criterion = nn.MSELoss()

    # baseline
    model = cpm.CPM(k=15)
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    train_model(training_dataset_path=training_dataset_path,
                val_data_path=val_data_path, model=model,
                criterion=criterion, optimizer=optimizer,
                max_epoch=300, batch_size=1, save_name='cpm_atrw')

    # transfer learning with lsp pretrained parameter
    model = transfer_model('model/best_cpm.pth', model)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    train_model(training_dataset_path=training_dataset_path,
                val_data_path=val_data_path, model=model,
                criterion=criterion, optimizer=optimizer,
                max_epoch=300, batch_size=1, save_name='cpm_atrw_transfer')

    # baseline of depthwise pointwise convolution
    model = cpm_condense.CPM_dpc(k=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    train_model(training_dataset_path=training_dataset_path,
                val_data_path=val_data_path, model=model,
                criterion=criterion, optimizer=optimizer,
                max_epoch=300, batch_size=1, save_name='cpm_atrw_dpc')

    # teacher-student mode to distillate knowledge (condense network)
    try:
        model_teacher = torch.load('model\\best_cpm_atrw_transfer.pth')
    except:
        print('Well trained teacher model is not ready yet.')
    model = cpm_condense.CPM_dpc(k=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    train_model(training_dataset_path=training_dataset_path,
                val_data_path=val_data_path, model=model,
                criterion=criterion, optimizer=optimizer,
                max_epoch=300, batch_size=1, save_name='cpm_atrw_dpc_ts')
    print('origin model size:')
    print_size_of_model(model_teacher)
    print('condesed model size:')
    print_size_of_model(model)
    '''
    # post-training Qutization model
    qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    best_model_path = os.path.join(
        os.getcwd(), 'model\\best_cpm_atrw_dpc_ts.pth')
    save_model_path = os.path.join(
        os.getcwd(), 'model\\best_cpm_atrw_dpc_ts_quant.pth')
    model = cpm_condense.CPM_dpc_quant(k=15)
    model1 = transfer_model(best_model_path, model)
    model1.qconfig = qconfig
    model1_prepared = torch.quantization.prepare(model1)
    model1_prepared.eval()
    data = LSP_DATA('lsp', training_dataset_path, 8,
                    Compose([RandomResized(), RandomCrop(368)]))
    train_loader = torch.utils.data.dataloader.DataLoader(
        data, batch_size=4)
    print(len(train_loader))
    for j, data in tqdm(enumerate(train_loader)):
        inputs, heatmap, centermap, _ = data

        inputs = inputs
        heatmap = heatmap
        centermap = centermap

        input_var = torch.autograd.Variable(inputs)
        heatmap_var = torch.autograd.Variable(heatmap)
        centermap_var = torch.autograd.Variable(centermap)

        heat1, heat2, heat3, heat4, heat5, heat6 = model1_prepared(
            input_var, centermap_var)
    model1_prepared_int8 = torch.quantization.convert(model1_prepared)
    print('origin model size:')
    print_size_of_model(model1)
    print('qunatized model size:')
    print_size_of_model(model1_prepared_int8)
    torch.save(model1_prepared_int8, save_model_path)
