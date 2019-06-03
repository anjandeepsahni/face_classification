import os
import csv
import time
import torch
import argparse
from model import *
from dataset import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Paths
MODEL_PATH = './../Models'
TEST_RESULT_PATH = './../Results'
TEST_VRFN_FILE = './../Data/test_trials_verification_student_new.txt'

# Defaults
DEFAULT_RUN_MODE = 'train'
DEFAULT_TASK = 'classify'
DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_TEST_BATCH_SIZE = 500
DEFAULT_NUM_CLASSES = 2300

# Hyperparameters.
LEARNING_RATE = 1e-2
LEARNING_RATE_STEP = 3
LEARNING_RATE_DECAY = 0.1
WEIGHT_DECAY = 5e-5
WARM_UP_EPOCHS = 3

# Models for ensembling.
MNV2_V1_W1 = './../Models/model_20190304-023557_val_69.246.pt'
MNV2_V1_W2 = './../Models/model_20190304-033928_val_69.028.pt'
MNV2_V1_W3 = './../Models/model_20190304-035521_val_69.159.pt'
MNV2_V1_W4 = './../Models/model_20190304-041113_val_69.050.pt'
MNV2_V1_W5 = './../Models/model_20190304-042705_val_69.202.pt'

MNV2_V2_W1 = './../Models/model_20190309-043550_val_68.768.pt'
MNV2_V2_W2 = './../Models/model_20190309-050314_val_68.681.pt'
MNV2_V2_W3 = './../Models/model_20190309-050758_val_68.746.pt'
MNV2_V2_W4 = './../Models/model_20190309-051241_val_68.463.pt'
MNV2_V2_W5 = './../Models/model_20190309-051720_val_68.811.pt'

MNV2_V3_W1 = './../Models/model_20190309-221316_val_66.355.pt'
MNV2_V3_W2 = './../Models/model_20190309-223033_val_66.442.pt'
MNV2_V3_W3 = './../Models/model_20190309-222905_val_66.464.pt'
MNV2_V3_W4 = './../Models/model_20190309-224337_val_66.486.pt'
MNV2_V3_W5 = './../Models/model_20190309-223917_val_66.594.pt'

RESNET50_W1 = './../Models/model_20190308-002758_val_65.181.pt'
RESNET50_W2 = './../Models/model_20190308-004053_val_65.225.pt'
RESNET50_W3 = './../Models/model_20190308-005346_val_65.290.pt'
RESNET50_W4 = './../Models/model_20190308-010639_val_65.486.pt'
RESNET50_W5 = './../Models/model_20190308-011932_val_64.899.pt'

# Initialize weights using xavier initialization.
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

# Saves test results to csv file for kaggle submission.
def save_test_results(predictions, ensemble=False):
    predictions = list(predictions.cpu().numpy())
    predictions_count = list(range(len(predictions)))
    csv_output = [[i,j] for i,j in zip(predictions_count,predictions)]
    if not ensemble:
        result_file_path = os.path.join(TEST_RESULT_PATH,\
                'result_{}.csv'.format((str.split(str.split(args.model_path, '/')[-1], '.pt')[0])))
    else:
        result_file_path = os.path.join(TEST_RESULT_PATH,\
                'result_ensemble_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S")))
    with open(result_file_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['id', 'label'])
        csv_writer.writerows(csv_output)

# Saves test results to csv file for kaggle submission (for face verification).
def save_test_verification_results(predictions, ensemble=False):
    predictions = list(predictions.cpu().numpy())
    with open(TEST_VRFN_FILE) as test_file:
        test_lines = test_file.readlines()
    predictions_title = [line.strip('\n') for line in test_lines]
    csv_output = [[i,j] for i,j in zip(predictions_title,predictions)]
    if not ensemble:
        result_file_path = os.path.join(TEST_RESULT_PATH,\
                'result_vrfn_{}.csv'.format((str.split(str.split(args.model_path, '/')[-1], '.pt')[0])))
    else:
        result_file_path = os.path.join(TEST_RESULT_PATH,\
                'result_vrfn_ensemble_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S")))
    with open(result_file_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['trial', 'score'])
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

def val_model_ensemble(models, val_loader, criterion, device):
    with torch.no_grad():
        for m in models:
            m.eval()
            m.to(device)
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.long().to(device)
            outputs = []
            predicted = []
            for idx, m in enumerate(models):
                outputs.append(m(data)[1])    # Only pick label output.
                p = F.softmax(outputs[idx], dim=1)
                predicted.append(p)
            net_prediction = predicted[0]
            for p in predicted[1:]:
                net_prediction += p
            net_prediction = net_prediction/(len(models))
            _, net_prediction = torch.max(net_prediction, 1)
            total_predictions += target.size(0)
            correct_predictions += torch.sum(torch.eq(net_prediction, target)).item()
            loss = 0
            for out in outputs:
                loss += criterion(out, target)
            loss = loss/(len(models))
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
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            all_predictions.append(predicted)
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)), end="\r", flush=True)
        # Join list of predicted tensors.
        all_predictions = torch.cat(all_predictions, 0)
        # Save predictions in csv file.
        save_test_results(all_predictions)
        end_time = time.time()
        print('\nTotal Test Predictions: %d Time: %d s' % (all_predictions.size()[0], end_time - start_time))

def test_model_ensemble(models, test_loader, device):
    with torch.no_grad():
        for m in models:
            m.eval()
            m.to(device)
        start_time = time.time()
        all_predictions = []
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            outputs = []
            predicted = []
            for idx, m in enumerate(models):
                outputs.append(m(data)[1])    # Only pick label output.
                p = F.softmax(outputs[idx], dim=1)
                predicted.append(p)
            net_prediction = predicted[0]
            for p in predicted[1:]:
                net_prediction += p
            net_prediction = net_prediction/(len(models))
            _, net_prediction = torch.max(net_prediction, 1)
            all_predictions.append(net_prediction)
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)), end="\r", flush=True)
        # Join list of predicted tensors.
        all_predictions = torch.cat(all_predictions, 0)
        # Save predictions in csv file.
        save_test_results(all_predictions, ensemble=True)
        end_time = time.time()
        print('\nTotal Test Predictions: %d Time: %d s' % (all_predictions.size()[0], end_time - start_time))

def val_model_verification(model, val_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        all_target = []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)  # For scoring img similarity.
        start_time = time.time()
        for batch_idx, (data1, data2, target) in enumerate(val_loader):
            data1, data2, target = data1.to(device), data2.to(device), target.long().to(device)
            output1 = model(data1)[0]    # Only pick embedding output.
            output2 = model(data2)[0]    # Only pick embedding output.
            sim_scores = cos(output1, output2)
            all_predictions.append(sim_scores)
            all_target.append(target)
            print('Validation Iteration: %d/%d' % (batch_idx+1, len(val_loader)), end="\r", flush=True)
        all_predictions = torch.cat(all_predictions, 0)
        all_target = torch.cat(all_target, 0)
        end_time = time.time()
        net_auc = roc_auc_score(all_target.cpu().numpy(), all_predictions.cpu().numpy())
        print('\nValidation Accuracy (AUC): %5.3f Time: %d s' % \
                (net_auc, end_time - start_time))
        return net_auc

def val_model_ensemble_verification(models, val_loader, criterion, device):
    with torch.no_grad():
        for m in models:
            m.eval()
            m.to(device)
        all_predictions = []
        all_target = []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)  # For scoring img similarity.
        start_time = time.time()
        for batch_idx, (data1, data2, target) in enumerate(val_loader):
            data1, data2, target = data1.to(device), data2.to(device), target.long().to(device)
            outputs1 = []
            outputs2 = []
            predicted = []
            for idx, m in enumerate(models):
                outputs1.append(m(data1)[0])    # Only pick embedding output.
                outputs2.append(m(data2)[0])    # Only pick embedding output.
                sim_scores = cos(outputs1[idx], outputs2[idx])
                predicted.append(sim_scores)
            all_predictions.append(sum(predicted)/len(predicted))
            all_target.append(target)
            print('Validation Iteration: %d/%d' % (batch_idx+1, len(val_loader)), end="\r", flush=True)
        all_predictions = torch.cat(all_predictions, 0)
        all_target = torch.cat(all_target, 0)
        end_time = time.time()
        net_auc = roc_auc_score(all_target.cpu().numpy(), all_predictions.cpu().numpy())
        print('\nValidation Accuracy (AUC): %5.3f Time: %d s' % \
                (net_auc, end_time - start_time))
        return net_auc

def test_model_verification(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        all_predictions = []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)  # For scoring img similarity.
        start_time = time.time()
        for batch_idx, (data1, data2) in enumerate(test_loader):
            data1, data2 = data1.to(device), data2.to(device)
            output1 = model(data1)[0]    # Only pick embedding output.
            output2 = model(data2)[0]    # Only pick embedding output.
            sim_scores = cos(output1, output2)
            all_predictions.append(sim_scores)
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)), end="\r", flush=True)
        # Join list of predicted tensors.
        all_predictions = torch.cat(all_predictions, 0)
        # Save predictions in csv file.
        save_test_verification_results(all_predictions)
        end_time = time.time()
        print('\nTotal Test Predictions: %d Time: %d s' % (all_predictions.size()[0], end_time - start_time))

def test_model_ensemble_verification(models, test_loader, device):
    with torch.no_grad():
        for m in models:
            m.eval()
            m.to(device)
        all_predictions = []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)  # For scoring img similarity.
        start_time = time.time()
        for batch_idx, (data1, data2) in enumerate(test_loader):
            data1, data2 = data1.to(device), data2.to(device)
            outputs1 = []
            outputs2 = []
            predicted = []
            for idx, m in enumerate(models):
                outputs1.append(m(data1)[0])    # Only pick embedding output.
                outputs2.append(m(data2)[0])    # Only pick embedding output.
                sim_scores = cos(outputs1[idx], outputs2[idx])
                predicted.append(sim_scores)
            all_predictions.append(sum(predicted)/len(predicted))
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)), end="\r", flush=True)
        # Join list of predicted tensors.
        all_predictions = torch.cat(all_predictions, 0)
        # Save predictions in csv file.
        save_test_verification_results(all_predictions, ensemble=True)
        end_time = time.time()
        print('\nTotal Test Predictions: %d Time: %d s' % (all_predictions.size()[0], end_time - start_time))

def parse_args():
    parser = argparse.ArgumentParser(description='Training/testing for Face Classifier/Verification.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default=DEFAULT_RUN_MODE, help='\'train\' or \'test\' mode.')
    parser.add_argument('--task', type=str, choices=['classify', 'verify'], default=DEFAULT_TASK, help='\'classify\' or \'verify\' mode.')
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=DEFAULT_TEST_BATCH_SIZE, help='Testing batch size.')
    parser.add_argument('--model_path', type=str, help='Path to model to be reloaded.')
    parser.add_argument('--model_ensemble', type=bool, default=False, help='True/false, if we have to model ensembling.')
    return parser.parse_args()

# For transfer learning.
def load_weights(model, model_path, device):
    pretrained_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    model_params = len(model_dict)
    pretrained_params = len(pretrained_dict)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    skip_count = 0
    bad_keys = [key for key in pretrained_dict.keys() if pretrained_dict[key].size() != model_dict[key].size()]
    for key in bad_keys:
        del pretrained_dict[key]
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print('Loaded model:', model_path)
    print('Skipped %d/%d params from pretrained for %d params in model.' \
            % (len(bad_keys), pretrained_params, model_params))
    return model

if __name__ == "__main__":
    # Create arg parser.
    args = parse_args()
    print('='*20)
    print('Input arguments:\n%s' % (args))

    # Validate arguments.
    if args.mode == 'test' and args.model_path == None and not args.model_ensemble:
        raise ValueError("Input Argument Error: Test mode specified but model_path is %s." % (args.model_path))

    # Instantiate face classification dataset.
    if args.task == 'classify':
        faceTrainDataset = FaceClassificationDataset(DEFAULT_NUM_CLASSES, mode='train')
        faceValDataset = FaceClassificationDataset(DEFAULT_NUM_CLASSES, mode='val')
        faceTestDataset = FaceClassificationDataset(DEFAULT_NUM_CLASSES, mode='test')
    else:
        faceTrainDataset = FaceVerificationDataset(DEFAULT_NUM_CLASSES, mode='train')
        faceValDataset = FaceVerificationDataset(DEFAULT_NUM_CLASSES, mode='val')
        faceTestDataset = FaceVerificationDataset(DEFAULT_NUM_CLASSES, mode='test')

    train_loader = DataLoader(faceTrainDataset, batch_size=args.train_batch_size,
                                shuffle=True, num_workers=8, )
    val_loader = DataLoader(faceValDataset, batch_size=args.train_batch_size,
                            shuffle=False, num_workers=8)
    test_loader = DataLoader(faceTestDataset, batch_size=args.test_batch_size,
                        shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_num_classes = faceTrainDataset.num_classes

    if args.model_ensemble:
        model_paths = [
            MNV2_V1_W1,
            MNV2_V1_W2,
            MNV2_V1_W3,
            MNV2_V1_W4,
            MNV2_V1_W5,
            MNV2_V2_W1,
            MNV2_V2_W2,
            MNV2_V2_W3,
            MNV2_V2_W4,
            MNV2_V2_W5,
            RESNET50_W1,
            RESNET50_W2,
            RESNET50_W3,
            RESNET50_W4,
            RESNET50_W5,
            MNV2_V3_W1,
            MNV2_V3_W2,
            MNV2_V3_W3,
            MNV2_V3_W4,
            MNV2_V3_W5
        ]
        models = [
            MobileNetV2_v1(model_num_classes),
            MobileNetV2_v1(model_num_classes),
            MobileNetV2_v1(model_num_classes),
            MobileNetV2_v1(model_num_classes),
            MobileNetV2_v1(model_num_classes),
            MobileNetV2_v2(model_num_classes),
            MobileNetV2_v2(model_num_classes),
            MobileNetV2_v2(model_num_classes),
            MobileNetV2_v2(model_num_classes),
            MobileNetV2_v2(model_num_classes),
            Resnet50(model_num_classes),
            Resnet50(model_num_classes),
            Resnet50(model_num_classes),
            Resnet50(model_num_classes),
            Resnet50(model_num_classes),
            MobileNetV2_v3(model_num_classes),
            MobileNetV2_v3(model_num_classes),
            MobileNetV2_v3(model_num_classes),
            MobileNetV2_v3(model_num_classes),
            MobileNetV2_v3(model_num_classes)
        ]
        for idx, m in enumerate(models):
            m.load_state_dict(torch.load(model_paths[idx], map_location=device))
            print('Loaded model:', model_paths[idx])
    else:
        model = FaceClassifier(model_num_classes)
        model.to(device)
        print('='*20)
        summary(model, input_size=(3, 32, 32))

    print("Running on device = %s." % (device))

    criterion = nn.CrossEntropyLoss()
    if not args.model_ensemble:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = LEARNING_RATE_DECAY)
        model.apply(init_weights)
        if args.model_path != None:
            model = load_weights(model, args.model_path, device)
            #model.load_state_dict(torch.load(args.model_path, map_location=device))
            #print('Loaded model:', args.model_path)

    n_epochs = 50
    print('='*20)

    if args.model_ensemble:
        if args.mode == 'train':
            # Only validate in ensemble mode.
            if args.task == 'classify':
                val_loss, val_acc = val_model_ensemble(models, val_loader, criterion, device)
            else:
                val_auc = val_model_ensemble_verification(models, val_loader, criterion, device)
        else:
            if args.task == 'classify':
                test_model_ensemble(models, test_loader, device)
            else:
                test_model_ensemble_verification(models, test_loader, device)
            print('='*20)
    else:
        if args.mode == 'train':
            if args.task == 'classify':
                for epoch in range(n_epochs):
                    print('Epoch: %d/%d' % (epoch+1,n_epochs))
                    train_loss = train_model(model, train_loader, criterion, optimizer, device)
                    val_loss, val_acc = val_model(model, val_loader, criterion, device)
                    # Checkpoint the model after each epoch.
                    finalValAcc = '%.3f'%(Val_acc[-1])
                    model_path = os.path.join(MODEL_PATH, 'model_{}_val_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), finalValAcc))
                    torch.save(model.state_dict(), model_path)
                    print('='*20)
                    if epoch >= WARM_UP_EPOCHS:
                        scheduler.step()
            else:
                val_auc = val_model_verification(model, val_loader, criterion, device)
        else:
            # Only testing the model.
            if args.task == 'classify':
                test_model(model, test_loader, device)
            else:
                test_model_verification(model, test_loader, device)
            print('='*20)
