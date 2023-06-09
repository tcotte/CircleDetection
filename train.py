import argparse
import math
import os
import time

import albumentations as A
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

from logger import WeightandBiaises
from losses import DIoULoss
from metrics import batch_iou
from model.object_detector import ObjectDetector
from torch_datasets import CustomImageDataset
from utils import create_directory

parser = argparse.ArgumentParser(description='Train custom model which enables to detect circles')
parser.add_argument('--epochs', type=int, default=200, help='total training epochs')
parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
parser.add_argument('--name', type=str, default=None, help='save to project/name')
parser.add_argument('--bbox-ratio', type=float, default=0.6, help='Bbox regression ratio on loss weight')
parser.add_argument('--prob-ratio', type=float, default=0.4, help='Probability that bbox is not a FP ratio on loss '
                                                                  'weight')
parser.add_argument('--unfreeze', type=int, default=5, help='Unfrozen layers')
parser.add_argument('--pretrained', action='store_true', help='Use backbone pretrained weights')
parser.add_argument('--wb', action='store_true', help='Visualize training with W&B API')
parser.add_argument('--imgsz', nargs='+', type=int, help="Image size used for the training")
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--path', type=str, required=True, help='Dataset root path')
parser.add_argument('--translation', action='store_true', help='Use translation augmentations')

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = args.lr
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
# specify the loss weights
LABELS = 1.0
BBOX = args.bbox_ratio
PROB = args.prob_ratio
UNFROZEN_LAYERS = args.unfreeze
PRETRAINED_BACKBONE = args.pretrained
# IMGSZ = (128, 128) # height, width
IMGSZ = args.imgsz # height, width


wb_visu = args.wb

model_name = args.name

CLASSES = ["Circle"]
bbox_format = 'albumentations'

list_train_transformation = [
        A.Equalize(mode='cv', by_channels=True, mask=None, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.Rotate(limit=90, p=0.5, border_mode=0, rotate_method="ellipse"),
        A.CLAHE(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
        ], p=0.0),
        A.Normalize(always_apply=True),

        A.augmentations.geometric.resize.Resize(*IMGSZ, interpolation=1, always_apply=False, p=1),
        ToTensorV2()]

if args.translation:
    list_train_transformation.insert(A.augmentations.geometric.transforms.Affine(scale=(0.5, 1), translate_percent=(0.15, 0.5), keep_ratio=True, p=0.5), 0)


train_transform = A.Compose(
    list_train_transformation,
    bbox_params=A.BboxParams(format=bbox_format, label_fields=['category_ids']),
)

test_transform = A.Compose([
    # A.augmentations.geometric.resize.Resize(*IMGSZ, interpolation=1, always_apply=False, p=1),
    A.Normalize(always_apply=True),
    ToTensorV2()],
    bbox_params=A.BboxParams(format=bbox_format, label_fields=['category_ids'])
)

train_dataset = CustomImageDataset(
    img_dir=os.path.join(args.path, "train/img"),
    label_dir=os.path.join(args.path, "train/labels"),
    transform=train_transform)

val_dataset = CustomImageDataset(
    img_dir=os.path.join(args.path, "val/img"),
    label_dir=os.path.join(args.path, "val/labels"),
    transform=test_transform)

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":

    print("[INFO] total training samples: {}...".format(len(train_dataset)))
    print("[INFO] total test samples: {}...".format(len(val_dataset)))
    # calculate steps per epoch for training and validation set
    train_steps = math.ceil(len(train_dataset) / BATCH_SIZE)
    val_steps = math.ceil(len(val_dataset) / BATCH_SIZE)
    # create data loaders
    # trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
    #                          shuffle=True, num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)
    # testLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
    #                         num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)

    # Network
    # load the ResNet network
    backbone = resnet18(pretrained=PRETRAINED_BACKBONE)
    # freeze some ResNet layers so they will *not* be updated during the training process
    params = backbone.state_dict()
    list_layers = list(params.keys())
    for name, param in backbone.named_parameters():
        if name in list_layers[-UNFROZEN_LAYERS:]:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # create our custom object detector model and flash it to the current
    # device
    objectDetector = ObjectDetector(backbone, len(CLASSES))
    objectDetector = objectDetector.to(DEVICE)
    # define our loss functions

    classLossFunc = CrossEntropyLoss()
    probLossFunc = BCELoss()

    bboxLossFunc = DIoULoss()
    # initialize the optimizer, compile the model, and show the model
    # summary
    opt = optim.Adam(objectDetector.parameters(), lr=INIT_LR)

    # initialize a dictionary to store training history
    H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
         "val_class_acc": [], "train_iou": [], "val_iou": []}

    if wb_visu:
        w_b = WeightandBiaises(project_name="circle_detection", run_id=model_name, interval_display=50, cfg={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": INIT_LR,
            "optimizer": repr(opt).split(" ")[0],
            "unfrozen_layers": UNFROZEN_LAYERS,
            "backbone_architecture": repr(backbone).split("(")[0],
            "pretrained_bacbone": PRETRAINED_BACKBONE,
            "dataset": "colored_circles",
            "weight_loss_bbox_regression": BBOX,
            "weight_loss_prob_x": PROB,
            "image_size": IMGSZ,
            "translation_aug": args.translation
        })
    else:
        w_b = None

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        # set the model in training mode
        objectDetector.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        bbox_train_loss = 0
        bbox_test_loss = 0

        obj_train_loss = 0
        obj_test_loss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0

        train_iou = 0
        val_iou = 0

        # loop over the training set
        for (images, labels, bboxes, probs, filenames) in training_loader:
            # send the input to the device
            labels = torch.Tensor(labels)
            # bboxes = torch.stack(bboxes, dim=1)

            bboxes = bboxes.to(torch.float32)
            bboxes = torch.squeeze(bboxes, 1)

            (images, labels, bboxes, probs) = (images.to(DEVICE),
                                        labels.to(DEVICE), bboxes.to(DEVICE), probs.to(DEVICE))
            # perform a forward pass and calculate the training loss
            opt.zero_grad()
            predictions = objectDetector(images)
            bbox_loss = bboxLossFunc(predictions[0], bboxes)
            objectness_loss = probLossFunc(predictions[2], probs.float())
            totalLoss = BBOX * bbox_loss + objectness_loss * PROB

            totalLoss = totalLoss.to(torch.float)

            # zero out the gradients, perform the backpropagation step,
            # and update the weights

            totalLoss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            train_iou += batch_iou(a=predictions[0].detach().cpu().numpy(), b=bboxes.cpu().numpy()).sum() / len(bboxes)
            totalTrainLoss += totalLoss
            bbox_train_loss += bbox_loss
            obj_train_loss += objectness_loss
            trainCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            objectDetector.eval()
            # loop over the validation set
            for (images, labels, bboxes, probs, _) in validation_loader:
                # send the input to the device
                labels = torch.Tensor(labels)
                bboxes = torch.squeeze(bboxes, 1)

                # bboxes = torch.stack(bboxes, dim=1)
                (images, labels, bboxes, probs) = (images.to(DEVICE),
                                            labels.to(DEVICE), bboxes.to(DEVICE), probs.to(DEVICE))
                # make the predictions and calculate the validation loss
                predictions = objectDetector(images)

                bbox_loss = bboxLossFunc(predictions[0], bboxes.to(torch.float32))
                # classLoss = classLossFunc(predictions[1], labels)
                objectness_loss = probLossFunc(predictions[2], probs.float())
                totalLoss = BBOX * bbox_loss + objectness_loss * PROB
                totalValLoss += totalLoss
                bbox_test_loss += bbox_loss
                obj_test_loss += objectness_loss
                # calculate the number of correct predictions
                val_iou += batch_iou(a=predictions[0].detach().cpu().numpy(), b=bboxes.cpu().numpy()).sum() / len(
                    bboxes)
                valCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()

                if w_b is not None:
                    w_b.plot_one_batch(predictions[0], images, [None]*len(predictions[0]), e)

        # calculate the average training and validation loss
        avg_train_loss = totalTrainLoss / train_steps
        avg_val_loss = totalValLoss / val_steps


        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(train_dataset)
        valCorrect = valCorrect / len(val_dataset)
        # update our training history
        H["total_train_loss"].append(avg_train_loss.cpu().detach().numpy())
        H["train_class_acc"].append(trainCorrect)
        H["total_val_loss"].append(avg_val_loss.cpu().detach().numpy())
        H["val_class_acc"].append(valCorrect)
        H["train_iou"].append(train_iou / train_steps)
        H["val_iou"].append(val_iou / val_steps)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.8f}".format(
            avg_train_loss, train_iou / train_steps))
        print("Val loss: {:.6f}, Val accuracy: {:.8f}".format(
            avg_val_loss, val_iou / val_steps))
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))

        if w_b is not None:
            if args.prob_ratio > 0:
                w_b.log_detailed_stats(stats={
                    "train_accuracy": train_iou / train_steps, "test_accuracy": val_iou / val_steps,
                    "train_loss": avg_train_loss.cpu().detach().numpy(), "test_loss": avg_val_loss.cpu().detach().numpy(),
                    # To complete
                    "train_obj_accuracy": 0, "test_obj_accuracy": 0,
                    "train_bbox_accuracy": train_iou / train_steps, "test_bbox_accuracy": val_iou / val_steps,
                    "train_obj_loss": obj_train_loss / train_steps, "test_obj_loss": obj_test_loss / val_steps,
                    "train_bbox_loss": bbox_train_loss, "test_bbox_loss": bbox_test_loss
                })
            else:
                w_b.log_accuracy(train_accuracy=train_iou / train_steps, test_accuracy=val_iou / val_steps, epoch=e,
                                 commit=False)
                w_b.log_losses(train_loss=avg_train_loss.cpu().detach().numpy(),
                               test_loss=avg_val_loss.cpu().detach().numpy(), epoch=e, commit=True)
                w_b.log_table(e)


    # serialize the model to disk
    print("[INFO] saving object detector model...")
    # torch.save(objectDetector, MODEL_PATH)
    # serialize the label encoder to disk
    print("[INFO] saving label encoder...")
    output_path = "trained_models"
    create_directory(output_path)
    torch.save(objectDetector, os.path.join(output_path, model_name + '.pt'))
    if w_b is not None:
        w_b.save_model(model_path=os.path.join(output_path, model_name + ".pth"))
    print("[INFO] Model was saved online")
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
