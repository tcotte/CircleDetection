import math
import os
import time

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

from metrics import batch_iou
from model.object_detector import ObjectDetector
from torch_datasets import CustomImageDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-3
NUM_EPOCHS = 50
BATCH_SIZE = 64
# specify the loss weights
LABELS = 1.0
BBOX = 1.0

model_name = "50_epochs_giouloss"

CLASSES = ["Circle"]

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

train_dataset = CustomImageDataset(
    img_dir=r"datasets/dataset_circle/train/img",
    label_dir=r"datasets/dataset_circle/train/labels",
    transform=transforms)

val_dataset = CustomImageDataset(
    img_dir=r"datasets/dataset_circle/val/img",
    label_dir=r"datasets/dataset_circle/val/labels",
    transform=transforms)

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    # for i in range(10):
    #     img, label = dataset[i]
    #     plt.imshow(img.permute(1, 2, 0))
    #     print(label)
    #     plt.show()

    path_img = r"datasets/dataset_circle/train/img"
    path_labels = r"datasets/dataset_circle/train/labels"

    print("[INFO] total training samples: {}...".format(len(train_dataset)))
    print("[INFO] total test samples: {}...".format(len(val_dataset)))
    # calculate steps per epoch for training and validation set
    trainSteps = math.ceil(len(train_dataset) / BATCH_SIZE)
    valSteps = math.ceil(len(val_dataset) / BATCH_SIZE)
    # create data loaders
    trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)
    testLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)

    # Network
    # load the ResNet50 network
    resnet = resnet50(pretrained=True)
    # freeze all ResNet50 layers so they will *not* be updated during the
    # training process
    for param in resnet.parameters():
        param.requires_grad = False

    # create our custom object detector model and flash it to the current
    # device
    objectDetector = ObjectDetector(resnet, len(CLASSES))
    objectDetector = objectDetector.to(DEVICE)
    # define our loss functions

    classLossFunc = CrossEntropyLoss()

    bboxLossFunc = MSELoss()
    # initialize the optimizer, compile the model, and show the model
    # summary
    opt = optim.Adam(objectDetector.parameters(), lr=INIT_LR)

    # initialize a dictionary to store training history
    H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
         "val_class_acc": [], "train_iou": [], "val_iou": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        # set the model in training mode
        objectDetector.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0

        train_iou = 0
        val_iou = 0

        # loop over the training set
        for (images, labels, bboxes, filenames) in trainLoader:
            # send the input to the device
            labels = torch.Tensor(labels)
            # bboxes = torch.stack(bboxes, dim=1)

            bboxes = bboxes.to(torch.float32)

            (images, labels, bboxes) = (images.to(DEVICE),
                                        labels.to(DEVICE), bboxes.to(DEVICE))
            # perform a forward pass and calculate the training loss
            opt.zero_grad()
            predictions = objectDetector(images)
            bboxLoss = bboxLossFunc(predictions[0], bboxes)

            classLoss = classLossFunc(predictions[1], labels)
            totalLoss = BBOX * bboxLoss
            totalLoss = totalLoss.to(torch.float)

            # zero out the gradients, perform the backpropagation step,
            # and update the weights

            totalLoss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            train_iou += batch_iou(a=predictions[0].detach().cpu().numpy(), b=bboxes.cpu().numpy()).sum() / len(bboxes)
            totalTrainLoss += totalLoss
            trainCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            objectDetector.eval()
            # loop over the validation set
            for (images, labels, bboxes, _) in testLoader:
                # send the input to the device
                labels = torch.Tensor(labels)
                # bboxes = torch.stack(bboxes, dim=1)
                (images, labels, bboxes) = (images.to(DEVICE),
                                            labels.to(DEVICE), bboxes.to(DEVICE))
                # make the predictions and calculate the validation loss
                predictions = objectDetector(images)
                bboxLoss = bboxLossFunc(predictions[0], bboxes)
                classLoss = classLossFunc(predictions[1], labels)
                totalLoss = BBOX * bboxLoss
                totalValLoss += totalLoss
                # calculate the number of correct predictions
                val_iou += batch_iou(a=predictions[0].detach().cpu().numpy(), b=bboxes.cpu().numpy()).sum() / len(
                    bboxes)
                valCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(train_dataset)
        valCorrect = valCorrect / len(val_dataset)
        # update our training history
        H["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_class_acc"].append(trainCorrect)
        H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_class_acc"].append(valCorrect)
        H["train_iou"].append(train_iou/trainSteps)
        H["val_iou"].append(val_iou/valSteps)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.8f}".format(
            avgTrainLoss, train_iou/trainSteps))
        print("Val loss: {:.6f}, Val accuracy: {:.8f}".format(
            avgValLoss, val_iou/valSteps))
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))

    # serialize the model to disk
    print("[INFO] saving object detector model...")
    # torch.save(objectDetector, MODEL_PATH)
    # serialize the label encoder to disk
    print("[INFO] saving label encoder...")
    torch.save(objectDetector, os.path.join("trained_models", model_name + '.pt'))
    # f = open(LE_PATH, "wb")
    # f.write(pickle.dumps(le))
    # f.close()
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["total_train_loss"], label="total_train_loss")
    plt.plot(H["total_val_loss"], label="total_val_loss")
    plt.plot(H["train_iou"], label="train_acc_iou")
    plt.plot(H["val_iou"], label="val_acc_iou")
    plt.title("Total Training Loss and Classification Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    # save the training plot
    plt.show()