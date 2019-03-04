import os
import csv
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import FaceClassifier
from model import CenterLoss
from torch.utils.data import DataLoader
from dataset import FaceClassificationDataset

# Paths
MODEL_PATH = './../Models'
TEST_RESULT_PATH = './../Results'

# Defaults
DEFAULT_RUN_MODE = 'train'
DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_TEST_BATCH_SIZE = 128
DEFAULT_NUM_CLASSES = 2300

# Hyperparameters.
LEARNING_RATE = 1e-2
LEARNING_RATE_CENTER = 0.5
LEARNING_RATE_STEP = 0.7
WEIGHT_DECAY = 5e-5
CLOSS_WEIGHT = 1
WARM_UP_EPOCHS = 5

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

def save_test_results(predictions):
    predictions = list(predictions.cpu().numpy())
    predictions_count = list(range(len(predictions)))
    csv_output = [[i,j] for i,j in zip(predictions_count,predictions)]
    result_file_path = os.path.join(TEST_RESULT_PATH,\
            'result_{}.csv'.format((str.split(str.split(args.model_path, '/')[-1], '.pt')[0])))
    with open(result_file_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['id', 'label'])
        csv_writer.writerows(csv_output)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    model.to(device)
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.long().to(device)
        outputs = model(data)[1]    # Only pick label output.
        loss = criterion(outputs, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        print('Train Iteration: %d/%d Loss = %5.4f' % \
                (batch_idx+1, len(train_loader), (running_loss/(batch_idx+1))), \
                end="\r", flush=True)
    end_time = time.time()
    running_loss /= len(train_loader)
    print('\nTraining Loss: %5.4f Time: %d s' % (running_loss, end_time - start_time))
    return running_loss

def train_model_closs(model, train_loader, criterion_label, criterion_closs, optimizer_label, optimizer_closs, device):
    model.train()
    model.to(device)
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer_label.zero_grad()
        optimizer_closs.zero_grad()
        data, target = data.to(device), target.to(device)
        features, outputs = model(data)
        l_loss = criterion_label(outputs, target.long())
        c_loss = criterion_closs(features, target.long())
        loss = l_loss + CLOSS_WEIGHT * c_loss
        loss.backward()
        optimizer_label.step()
       # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_closs.parameters():
            param.grad.data *= (1. / CLOSS_WEIGHT)
        optimizer_closs.step()
        running_loss += loss.item()
        print('Train Iteration: %d/%d Loss = %5.4f' % \
                (batch_idx+1, len(train_loader), (running_loss/(batch_idx+1))), \
                end="\r", flush=True)
    end_time = time.time()
    running_loss /= len(train_loader)
    print('\nTraining Loss: %5.4f Time: %d s' % (running_loss, end_time - start_time))
    return running_loss

def val_model(model, val_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.long().to(device)
            outputs = model(data)[1]    # Only pick label output.
            #_, predicted = torch.max(outputs.data, 1)
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            predicted = predicted.view(-1)
            total_predictions += target.size(0)
            correct_predictions += torch.sum(torch.eq(predicted, target)).item()
            loss = criterion(outputs, target)
            running_loss += loss.item()
            print('Validation Iteration: %d/%d Loss = %5.4f' % \
                    (batch_idx+1, len(val_loader), (running_loss/(batch_idx+1))), \
                    end="\r", flush=True)
        end_time = time.time()
        running_loss /= len(val_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('\nValidation Loss: %5.4f Validation Accuracy: %5.3f Time: %d s' % \
                (running_loss, acc, end_time - start_time))
        return running_loss, acc

def val_model_closs(model, val_loader, criterion_label, criterion_closs, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.long().to(device)
            features, outputs = model(data)
            #_, predicted = torch.max(outputs.data, 1)
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            predicted = predicted.view(-1)
            total_predictions += target.size(0)
            correct_predictions += torch.sum(torch.eq(predicted, target)).item()
            l_loss = criterion_label(outputs, target)
            c_loss = criterion_closs(features, target)
            loss = l_loss + CLOSS_WEIGHT * c_loss
            running_loss += loss.item()
            print('Validation Iteration: %d/%d Loss = %5.4f' % \
                    (batch_idx+1, len(val_loader), (running_loss/(batch_idx+1))), \
                    end="\r", flush=True)
        end_time = time.time()
        running_loss /= len(val_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('\nValidation Loss: %5.4f Validation Accuracy: %5.3f Time: %d s' % \
                (running_loss, acc, end_time - start_time))
        return running_loss, acc

def test_model(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        start_time = time.time()
        all_predictions = []
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            outputs = model(data)[1]    # Only pick label output.
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted)
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)), end="\r", flush=True)
        # Join list of predicted tensors.
        all_predictions = torch.cat(all_predictions, 0)
        # Save predictions in csv file.
        save_test_results(all_predictions)
        end_time = time.time()
        print('\nTotal Test Predictions: %d Time: %d s' % (all_predictions.size()[0], end_time - start_time))

def test_model_verification(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        start_time = time.time()
        all_predictions = []
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            outputs = model(data)[0]    # Only pick label output.
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted)
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)), end="\r", flush=True)
        # Join list of predicted tensors.
        all_predictions = torch.cat(all_predictions, 0)
        # Save predictions in csv file.
        save_test_results(all_predictions)
        end_time = time.time()
        print('\nTotal Test Predictions: %d Time: %d s' % (all_predictions.size()[0], end_time - start_time))

def parse_args():
    parser = argparse.ArgumentParser(description='Training/testing for Face Classifier.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default=DEFAULT_RUN_MODE, help='\'train\' or \'test\' mode.')
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=DEFAULT_TEST_BATCH_SIZE, help='Testing batch size.')
    parser.add_argument('--reload_model', type=bool, default=False, help='True/false, if we have to reload a model.')
    parser.add_argument('--model_path', type=str, help='Path to model to be reloaded.')
    return parser.parse_args()

if __name__ == "__main__":
    # Create arg parser.
    args = parse_args()
    print('='*20)
    print('Input arguments:\n%s' % (args))

    # Validate arguments.
    if args.mode == 'test' and (not args.reload_model or args.model_path == None):
        raise ValueError("Input Argument Error: Test mode specified but reload_model is %s and model_path is %s." \
                        % (args.reload_model, args.model_path))

    if (args.reload_model and (args.model_path == None)):
        raise ValueError("Input Argument Error: Reload model specified true but model_path is %s." \
                        % (args.model_path))

    # Instantiate face classification dataset.
    faceTrainDataset = FaceClassificationDataset(DEFAULT_NUM_CLASSES, mode='train')
    faceValDataset = FaceClassificationDataset(DEFAULT_NUM_CLASSES, mode='val')
    faceTestDataset = FaceClassificationDataset(DEFAULT_NUM_CLASSES, mode='test')
    train_loader = DataLoader(faceTrainDataset, batch_size=args.train_batch_size,
                                shuffle=True, num_workers=8)
    val_loader = DataLoader(faceValDataset, batch_size=args.train_batch_size,
                            shuffle=False, num_workers=8)
    test_loader = DataLoader(faceTestDataset, batch_size=args.test_batch_size,
                        shuffle=False, num_workers=8)

    model_num_classes = faceTrainDataset.num_classes
    model = FaceClassifier(model_num_classes)
    print('='*20)
    print(model)
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Model Parameters:', model_total_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device = %s." % (device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = LEARNING_RATE_STEP)

    #criterion_label = nn.CrossEntropyLoss()
    #criterion_closs = CenterLoss(model_num_classes, 10, device)
    #optimizer_label = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)
    #optimizer_closs = optim.SGD(criterion_closs.parameters(), lr=LEARNING_RATE_CENTER)

    if args.reload_model:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('Loaded model:', args.model_path)
    else:
        model.apply(init_weights)

    n_epochs = 50
    Train_loss = []
    Val_loss = []
    Val_acc = []

    print('='*20)

    if args.mode == 'train':
        for i in range(n_epochs):
            print('Epoch: %d/%d' % (i+1,n_epochs))
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            #train_loss = train_model_closs(model, train_loader, criterion_label, criterion_closs, optimizer_label, optimizer_closs, device)
            val_loss, val_acc = val_model(model, val_loader, criterion, device)
            #val_loss, val_acc = val_model_closs(model, val_loader, criterion_label, criterion_closs, device)
            Train_loss.append(train_loss)
            Val_loss.append(val_loss)
            Val_acc.append(val_acc)
            # Checkpoint the model after each epoch.
            finalValAcc = '%.3f'%(Val_acc[-1])
            model_path = os.path.join(MODEL_PATH, 'model_{}_val_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), finalValAcc))
            torch.save(model.state_dict(), model_path)
            print('='*20)
            if i >= WARM_UP_EPOCHS:
                scheduler.step()
    else:
        # Only testing the model.
        test_model(model, test_loader, device)
        print('='*20)
