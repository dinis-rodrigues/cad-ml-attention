import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from transformers import AdamW
from torchvision import transforms
from PIL import Image
import os
import wandb
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from skimage import io, transform
import numpy as np
import pandas as pd
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.append("/media/medstorage2/Attention/cad-ml-attention/src/utils")

from cadDatasetLoader import TorchCoronaryDataset, f1_loss, RMSELoss

def model_accuraccy(predictions, target):
    # Calculate accuraccy
    _, predicted = torch.max(predictions, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    accuraccy = correct / total
    return accuraccy

def train(config,
          dataDir="",
          framesDir="",
          foldNum=1,
          num_batch=16,
          num_epochs=10,
          learning_rate=1e-3,
          optimizer_type="adamw"):
    # reset_wandb_env()
    # worker_data = worker_q.get()
    # run_name = "{}-{}".format(config["sweep_run_name"], foldNum)
    # config = worker_data.config

    train_csv = [f"{dataDir}/fold_1_train.csv",
                 f"{dataDir}/fold_2_train.csv",
                 f"{dataDir}/fold_3_train.csv",
                 f"{dataDir}/fold_4_train.csv",
                 f"{dataDir}/fold_5_train.csv"]

    test_csv = [f"{dataDir}/fold_1_test.csv",
                f"{dataDir}/fold_2_test.csv",
                f"{dataDir}/fold_3_test.csv",
                f"{dataDir}/fold_4_test.csv",
                f"{dataDir}/fold_5_test.csv"]


    BATCH_SIZE = num_batch
    # ------- CSV with image paths and iFR values ---------
    train_file = train_csv[foldNum]
    test_file = test_csv[foldNum]

    # ------- Data Generators ---------
    img_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    train_dataset = TorchCoronaryDataset(csv_file=train_file,
                                         image_processor=img_processor,
                                         root_dir=framesDir,
                                         classification=False)
    val_dataset = TorchCoronaryDataset(csv_file=test_file,
                                       image_processor=img_processor,
                                       root_dir=framesDir,
                                       classification=False)

    training_samples = len(train_dataset)
    validation_samples = len(val_dataset)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 1}
    validation_params = {'batch_size': 1,
                        'shuffle': False,
                        'num_workers': 1}

    training_generator = torch.utils.data.DataLoader(train_dataset,
                                                        **train_params)
    val_generator = torch.utils.data.DataLoader(val_dataset,
                                                **validation_params)
    print("Building model")
    # ------- Pre Trained Model ---------
    # Pre-trained model, with a new classification Head, classification and 2 labels
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Reaplace classification head
    model.classifier = torch.nn.Linear(in_features=768, out_features=1, bias=True)
    print("Model built")

    # Optimizer
    if optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        quit()

    num_training_steps = (training_samples // BATCH_SIZE)
    num_validation_steps = (validation_samples // 1)

    reduceLr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    'min',
    factor=0.2,
    patience=2,
    threshold=1e-5,
    verbose=True)

    # Loss function
    lossFunction = torch.nn.MSELoss()

    device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
    # device = torch.cuda.current_device()
    print("Using: ", device)
    print("Availabel count: ", torch.cuda.device_count())
    print("Current default device", torch.cuda.current_device())

    if torch.cuda.is_available():
        model.to(device)
    
    batch_losses = []
    print("Starting training")
    run = wandb.init(
        config=config,
        reinit=True,
        tags=["expCD"]
    )
    rmseLoss = RMSELoss()
    for epoch in range(num_epochs+1):  # loop over the dataset multiple times
        running_loss = 0.0
        running_rmse = 0.0
        running_acc = 0.0
        running_f1 = 0.0
        total_steps = 0
        loop = tqdm(enumerate(training_generator), total=num_training_steps)
        model.train()
        for step, batch in loop:
            inputs, target = batch["image"]["pixel_values"].to(device), batch["label"].to(device)
            target = torch.unsqueeze(target, 1).float()
            # print("Targets: ",target.size())
            # restart the gradients
            optimizer.zero_grad()

            #forward propagation through the network
            out = model(inputs)

            #calculate the loss
            predictions = out.logits.float()
            # print("Predictions:", predictions.size())
            loss = lossFunction(predictions, target)
            loss_rmse = rmseLoss(predictions, target)

            #track batch loss
            batch_losses.append(loss.item())
            running_loss += loss
            running_rmse += loss_rmse

            #backpropagation
            loss.backward()

            #update the parameters
            optimizer.step()

            # Calculate accuraccy
            predictions = (predictions>0.89).float().detach().cpu().numpy().tolist()
            target = (target > 0.89).float().detach().cpu().numpy().tolist()
            accuraccy = accuracy_score(target, predictions)
            running_acc += accuraccy
            f1_train_score = f1_score(target, predictions)
            running_f1 += f1_train_score

            # Update Progress bar
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), acc=accuraccy)
            total_steps += 1
        # Evaluate model after epoch
        with torch.no_grad():
            eval_loop = tqdm(enumerate(val_generator), total=num_validation_steps)
            model.eval()
            totalEval = 0
            correctEval = 0
            val_loss_track = 0
            val_rmse_track = 0
            val_steps=0
            predictedList = []
            targetList = []
            resTable=[]
            for step, batch in eval_loop:
                inputs, target = batch["image"]["pixel_values"].to(device), batch["label"].to(device)
                targetFloat = target.float()
                target = torch.unsqueeze(target, 1).float()
                out = model(inputs)

                #calculate the loss
                predictions = out.logits.float()
                loss = lossFunction(predictions, target)
                val_loss_track += loss
                val_rmse_loss = rmseLoss(predictions, target)
                val_rmse_track += val_rmse_loss

                # Calculate accuraccy, binarized
                predicted = (predictions > 0.89).float().detach().cpu().numpy().tolist()[0][0]
                predictedList.append(predicted)
                target = (target > 0.89).float().detach().cpu().numpy().tolist()
                targetList.append(target[0][0])
                totalEval += 1

                # calculate f1

                #Metrics for table
                resMetric = [batch["filename"][0]]
                # print("Eval pred", predictions)
                # print("Eval target", target)
                predStore = round(predictions.detach().cpu().numpy().tolist()[0][0],3)
                targetStore = targetFloat.detach().cpu().numpy().tolist()[0]
                
                resMetric.append(predStore)
                resMetric.append(round(targetStore,3))

                resTable.append(resMetric)




                val_steps+=1
            # Validation Metrics
            val_Acc = accuracy_score(targetList, predictedList)
            val_Loss = val_loss_track/ val_steps
            val_RMSE = val_rmse_track / val_steps
            valF1 = f1_score(targetList, predictedList)
            reduceLr.step(val_Loss)

            # Train metrics
            train_Acc = running_acc / total_steps
            train_f1 = running_f1 / total_steps
            train_Loss = running_loss / total_steps
            train_RMSE = running_rmse / total_steps

            run.log({"train_loss": train_Loss,
                     "train_acc": train_Acc,
                     "train_f1": train_f1,
                     "train_rmse": train_RMSE,
                     "val_loss": val_Loss,
                     "val_f1": valF1,
                     "val_rmse": val_RMSE,
                     "val_acc": val_Acc,
                     "epoch": epoch,
                     "predictions": wandb.Table(columns=["filename",
                                                         "Prediction",
                                                         "Target"],
                                                          data=resTable)
                        })

            print(f"Train Accuraccy: {train_Acc} / Train Loss: {train_Loss}" )
            print(f"Validation Accuraccy: {val_Acc} / Val Loss: {val_Loss}" )

    torch.save(model, "/media/medstorage2/Attention/models/vit_regCD224.pt")
    run.finish()
    # sweep_q.put(WorkerDoneData(val_acc=val_Acc))

def main():
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch_size",
                            help="Batch size",
                            default=16)
    parser.add_argument("-e", "--epochs",
                            help="Number of epochs",
                            default=10,)
    parser.add_argument("-l", "--learning_rate",
                            help="Learning rate",
                            default=1e-3,)
    parser.add_argument("-o", "--optimizer",
                            help="Optimizer",
                            default="adamw",)

    args = parser.parse_args()

    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    optimizer = args.optimizer


    print("Starting kamsd kmaskd maksd kajsn dkajnsd ")
    artery = "CD"
    dataDir = f"/media/medstorage2/Attention/cad-ml-attention/data/{artery}"
    framesDir = f"/media/medstorage/Data/Public/{artery}/frames"

    # Worker configuration
    num_folds = 5
    def check_mem(cuda_device):
        devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
        total, used = devices_info[int(cuda_device)].split(',')
        return total,used
    def occumpy_mem(cuda_device):
        total, used = check_mem(cuda_device)
        total = int(total)
        used = int(used)
        max_mem = int(total * 0.9)
        block_mem = max_mem - used
        x = torch.cuda.FloatTensor(256,1024,block_mem)
        del x
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # torch.cuda.set_device("cuda:1")
    occumpy_mem("0")

    config={"batch_size":int(args.batch_size),
    "epochs":int(args.epochs),
    "learning_rate":float(args.learning_rate),
    "optimizer":args.optimizer,}
    # "sweep_run_name":sweep_run_name,
    # "sweep_id":sweep_id}
    print("My config: ", config)

    # Initialize the sweep
    metrics = []
    # for num in range(num_folds):
    train(config,
        dataDir=dataDir,
        framesDir=framesDir,
        foldNum=1,
        num_batch=batch_size,
        num_epochs=epochs,
        learning_rate=learning_rate,
        optimizer_type=optimizer)

if __name__ == '__main__':
    main()

