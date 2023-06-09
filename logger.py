import os
from typing import Union, List, Any, Dict

import numpy as np
import wandb
from numpy import ndarray
import torch


class WeightandBiaises:
    def __init__(self, project_name: str = "my_dl_project", run_id=None, interval_display: int = 10,
                 cfg: Union[Dict, None] = None):
        """
        This class enables to send data from training to Weight&Biaises to visualize the
        behaviour of our training.
        :param project_name: name of our project on W&B
        :param run_id: name of our run in our project on W&B. If None, W&B will choose
        a random name for us.
        :param interval_display: this enables to display the mask_debugger with an interval
        of {interval_display} epochs.
        """
        if cfg is None:
            cfg = {}

        self.interval_display = interval_display
        self.class_labels = {
            0: "circle"
        }
        self.multiclass = True if len(self.class_labels) > 1 else False
        self.run_id = run_id

        wandb.login()
        wandb.init(id=run_id, project=project_name, config=cfg)
        self.run_id = wandb.run.name

        self.image_list = []

    @staticmethod
    def log_losses(train_loss: float, test_loss: float, epoch: int, commit: bool, **kwargs) -> None:
        """
        Log train and test losses in separate panels.
        :param test_loss:
        :param train_loss: average train loss for the current epoch.
        :param train_loss: average test loss for the current epoch.
        :param epoch: current epoch.
        """
        wandb.log({"Train/Total_Loss": train_loss, "Test/Total_Loss": test_loss}, step=epoch, commit=commit)

    @staticmethod
    def log_detailed_losses(train_obj_loss: float, test_obj_loss: float, train_bbox_loss: float,
                            test_bbox_loss: float, epoch: int, commit: bool, **kwargs) -> None:
        """
        Log train and test losses in separate panels.
        :param test_bbox_loss:
        :param train_bbox_loss:
        :param test_obj_loss:
        :param train_obj_loss:
        :param epoch: current epoch.
        """
        wandb.log({"Train/Objectness_Loss": train_obj_loss, "Test/Objectness_Loss": test_obj_loss,
                   "Train/Regression_Loss": train_bbox_loss, "Test/Regression_Loss": test_bbox_loss}, step=epoch,
                  commit=commit)


    def log_accuracy(self, train_accuracy: Union[None, float, List], test_accuracy: Union[float, List],
                     epoch: int, commit: bool, **kwargs) -> None:
        """
        Log iou accuracy.
        :param commit:
        :param train_accuracy:
        :param test_accuracy: average iou accuracy for the current epoch.
        :param epoch: current epoch.
        """
        if len(self.class_labels) < 2:
            if train_accuracy is not None:
                wandb.log({"Test/Total_Accuracy": test_accuracy, "Train/Total_Accuracy": train_accuracy}, step=epoch,
                          commit=False)
            else:
                wandb.log({"Test/Total_Accuracy": test_accuracy}, step=epoch, commit=commit)
        else:
            for idx, cls in enumerate(self.class_labels):
                wandb.log({"Test/Total_Accuracy_" + cls: test_accuracy[idx]}, commit=False)

    @staticmethod
    def log_detailed_accuracies(train_obj_accuracy: float, test_obj_accuracy: float, train_bbox_accuracy: float,
                                test_bbox_accuracy: float, epoch: int, commit: bool, **kwargs):
        wandb.log({"Train/Objectness_Accuracy": train_obj_accuracy, "Test/Objectness_Accuracy": test_obj_accuracy,
                   "Train/Regression_Accuracy": train_bbox_accuracy, "Test/Regression_Accuracy": test_bbox_accuracy},
                  step=epoch, commit=commit)

    def log_detailed_stats(self, stats: Dict):
        self.log_losses(**stats, commit=False)
        self.log_detailed_losses(**stats, commit=False)
        self.log_accuracy(**stats, commit=False)
        self.log_detailed_accuracies(**stats, commit=True)


    def visualize_one_image(self, pred, x) -> wandb.Image:
        """
        Transform mask prediction, ground_truth and image to wandb image which enables
        to display correctly the segmentation overlays.
        :param pred: prediction bounding box
        :param y: ground truth mask [1, H W]
        :param x: torch tensor image
        :return : wandb image which embeds the original picture, the prediction and the ground truth masks.
        """
        image = self.tensor2image(x)

        bboxes_data = []
        if len(pred.size()) == 1:
            pred = torch.unsqueeze(pred, 0)

        for box in pred:
            bboxes_data.append(
                {
                    "position": {
                        "minX": float(box[0].item()),
                        "maxX": float(box[2].item()),
                        "minY": float(box[1].item()),
                        "maxY": float(box[3].item())
                    },
                    "class_id": 0,
                    "box_caption": self.class_labels[0],
                    "scores": {
                        "acc": 1
                    }
                }
            )

        bbox_image = wandb.Image(image, boxes={
            "predictions": {
                "box_data": bboxes_data,
                "class_labels": self.class_labels
            },
        })
        return bbox_image

    def plot_one_batch(self, pred_batch, x_batch, y_batch,
                       e: int) -> None:
        """
        Get wandb Image for each batch in test loader and stock it.
        :param pred: prediction mask [B, 1, H W] where B is the batch size
        :param y: ground truth mask [B, 1, H W]
        :param x: torch tensor image [B, 3, H, W]
        :param e: current epoch
        """
        if (e != 0) and (e % self.interval_display == 0):
            for pred, x, y in zip(pred_batch, x_batch, y_batch):
                wandb_mask = self.visualize_one_image(pred, x)
                self.image_list.append(wandb_mask)

    def log_table(self, e: int):
        """
        Send the wandb images to W&B at the epoch's end.
        """
        if e % self.interval_display == 0:
            wandb.log({"localization_images_list": self.image_list})
        self.image_list = []

    @staticmethod
    def tensor2image(x) -> np.array:
        """
        Transform tensor to image numpy array.
        :param x: image float tensor [1, C, H, W]
        :return : image numpy array [H, W, C]
        """
        a = x.squeeze()
        a = a.permute(1, 2, 0)
        return a.detach().cpu().numpy()

    @staticmethod
    def tensors2masks(pred, y, confidence=0.5):
        """
        Transform prediction mask threshold at certain confidence to mask.
        :param: output model [1, H, W]
        :param: target mask [1, H, W]
        :return: prediction mask [H, W] and target mask [H, W]
        """
        target = y.squeeze().int()
        y_hat = torch.sigmoid(pred.squeeze())
        pred_mask = (y_hat > confidence).int()
        return pred_mask, target

    # @staticmethod
    # def pred2masks(predictions, conf_min: float = 0.35) -> ndarray:
    #     bkgd_mask = np.zeros(list(predictions.shape[1:]), dtype=int)
    #     for i, mask in enumerate(predictions):
    #         mask = torch.sigmoid(mask)
    #         mask = mask.cpu().numpy()
    #         mask = (mask > conf_min).astype(int)
    #         bkgd_mask += mask * (i + 1)
    #     return bkgd_mask
    #
    # @staticmethod
    # def target2mask(y: torch.FloatTensor, multiclass: bool = False) -> Union[ndarray, Any]:
    #     y = y.squeeze().int().detach().cpu().numpy()
    #     if not multiclass:
    #         return y
    #
    #     else:
    #         bkgd_mask = np.zeros(list(y.shape[1:]), dtype=int)
    #         for i, mask in enumerate(y):
    #             bkgd_mask += mask * (i + 1)
    #         return bkgd_mask

    # def save_model(self, model_path: str) -> None:
    #     """
    #     Save the last model into W&B
    #     :param model_path: location of model in local
    #     """
    #     artifact = wandb.Artifact('last_model', type='model')
    #     artifact.add_file(local_path=model_path, name="last.pth")
    #     wandb.log_artifact(artifact)

    def save_model(self, model_name: str, model) -> None:
        """
        Save the last model into W&B
        :param model_name: model's name
        """
        model.save(os.path.join(wandb.run.dir, model_name + ".pt"))
