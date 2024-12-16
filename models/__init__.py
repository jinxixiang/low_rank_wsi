import pytorch_lightning as pl
import torchmetrics.functional as tf
import importlib
import torch.nn.functional as F
import torch
from .model_utils import get_rank

# models import
from .ilra import ILRA
from .abmil import AbMIL
from .clam import CLAM_MB, CLAM_SB
from .dsmil import DSMIL


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class MILModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr_scheduler = None
        self.criterion = getattr(torch.nn, self.config["Loss"]["name"])(**self.config["Loss"]["params"])
        self.model = get_obj_from_str(config["Model"]["name"])(**config["Model"]["params"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # for CLAM_SB, CLAM_MB
        if self.config["Model"]["name"].split(".")[-1].startswith("CLAM"):
            logits, Y_prob, Y_hat, result_dict = self(batch)
            inst_loss = result_dict["instance_loss"]
            bag_loss = self.criterion(logits, y)
            loss = bag_loss + 0.3 * inst_loss
        # for DSMIL model
        elif self.config["Model"]["name"].split(".")[-1] == "DSMIL":
            logits, classes, Y_prob, Y_hat = self(x)
            max_prediction, index = torch.max(classes, 0)
            max_prediction = max_prediction.view(1, -1)
            loss_bag = self.criterion(logits, y)
            loss_max = self.criterion(max_prediction, y)
            loss = 0.5 * loss_bag + 0.5 * loss_max
        else:
            logits, Y_prob, Y_hat = self(x)
            loss = self.criterion(logits, y)

        return loss

    def on_train_epoch_end(self):
        self.lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.config["Model"]["name"].split(".")[-1].startswith("CLAM"):
            logits, Y_prob, Y_hat, result_dict = self(batch)
            inst_loss = result_dict["instance_loss"]
            bag_loss = self.criterion(logits, y)
            loss = bag_loss + 0.3 * inst_loss
        elif self.config["Model"]["name"].split(".")[-1] == "DSMIL":
            logits, classes, Y_prob, Y_hat = self(x)
            max_prediction, index = torch.max(classes, 0)
            max_prediction = max_prediction.view(1, -1)
            loss_bag = self.criterion(logits, y)
            loss_max = self.criterion(max_prediction, y)
            loss = 0.5 * loss_bag + 0.5 * loss_max
        else:
            logits, Y_prob, Y_hat = self(x)
            loss = self.criterion(logits, y)

        return {"loss": loss, "preds": Y_hat, "probs": Y_prob, "labels": y}

    def validation_step_end(self, batch_parts):
        # prediction form each GPU
        preds_gather = concat_all_gather(batch_parts["preds"])
        probs_gather = concat_all_gather(batch_parts["probs"])
        labels_gather = concat_all_gather(batch_parts["labels"])
        loss_gather = torch.mean(batch_parts["loss"])

        return {"loss": loss_gather,
                "preds": preds_gather,
                "probs": probs_gather,
                "labels": labels_gather}

    def validation_epoch_end(self, validation_step_outputs):
        # gather all validation results
        pred_list = torch.cat([out["preds"].detach() for out in validation_step_outputs], dim=0).squeeze(1)
        label_list = torch.cat([out["labels"].detach() for out in validation_step_outputs], dim=0)
        prob_list = torch.cat([out["probs"].detach() for out in validation_step_outputs], dim=0)

        eval_loss_dict = torch.stack([out["loss"].detach() for out in validation_step_outputs])
        eval_loss = torch.mean(eval_loss_dict)

        acc = tf.accuracy(pred_list, label_list, task="binary")
        auc = tf.auroc(prob_list[:, 1], label_list, task="binary")

        self.log("val_loss", eval_loss)
        self.log("val_auc", auc)

        if get_rank() == 0:
            print("validation prob_list shape", prob_list.shape)
            print(f"validation_performance acc: {acc:.4f}, auc: {auc:.4f} loss: {eval_loss: .4f}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.config["Model"]["name"].split(".")[-1].startswith("CLAM"):
            logits, Y_prob, Y_hat, result_dict = self(batch)
        elif self.config["Model"]["name"].split(".")[-1] == "DSMIL":
            logits, classes, Y_prob, Y_hat = self(x)
        else:
            logits, Y_prob, Y_hat = self(x)

        return {"preds": Y_hat, "probs": Y_prob, "labels": y}

    def test_step_end(self, batch_parts):
        # Generate predictions from each GPU, and subsequently aggregate all the results.
        preds_gather = concat_all_gather(batch_parts["preds"])
        probs_gather = concat_all_gather(batch_parts["probs"])
        labels_gather = concat_all_gather(batch_parts["labels"])

        return {"preds": preds_gather,
                "probs": probs_gather,
                "labels": labels_gather}

    def test_epoch_end(self, validation_step_outputs):
        pred_list = torch.cat([out["preds"].detach() for out in validation_step_outputs], dim=0).squeeze(1)
        label_list = torch.cat([out["labels"].detach() for out in validation_step_outputs], dim=0)
        prob_list = torch.cat([out["probs"].detach() for out in validation_step_outputs], dim=0)

        acc = tf.accuracy(pred_list, label_list, task="binary")
        auc = tf.auroc(prob_list[:, 1], label_list, task="binary")

        if get_rank() == 0:
            print(f"prob_list: {prob_list.shape}")
            print(f"acc: {acc:.4f} auc: {auc:.4f}")

    def configure_optimizers(self):
        conf_optim = self.config["Optimizer"]
        name = conf_optim["optimizer"]["name"]
        optimizer_cls = getattr(torch.optim, name)
        scheduler_cls = getattr(torch.optim.lr_scheduler, conf_optim["lr_scheduler"]["name"])
        optim = optimizer_cls(self.parameters(), **conf_optim["optimizer"]["params"])
        self.lr_scheduler = scheduler_cls(optim, **conf_optim["lr_scheduler"]["params"])
        return optim


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
