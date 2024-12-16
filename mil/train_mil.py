import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint
from wsi_dataset import WSIDataModule
import yaml
import importlib
from models import MILModel


def read_yaml(fpath):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return dict(yml)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


if __name__ == "__main__":
    # Configs
    fname = "config_clam_camelyon16_imagenet"
    config_yaml = read_yaml(f"./configs/{fname}.yaml")
    for key, value in config_yaml.items():
        print(f"{key.ljust(30)}: {value}")

    # run in ddp mode if num_gpus > 1
    num_gpus = 1
    dist = num_gpus > 1

    # create datamodule
    dm = WSIDataModule(config_yaml, split_k=0, dist=dist)  # split_k = [0,1,...9] for 10-fold cross-validation split

    resume_path = None

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = MILModel(config_yaml)
    checkpoint_cb = ModelCheckpoint(save_top_k=5,
                                    monitor="val_loss",
                                    mode="min",
                                    save_last=True,
                                    dirpath=f"./outputs/{fname}",
                                    filename="{epoch}-{val_auc:.3f}-{val_loss:.3f}")

    trainer = pl.Trainer(
        gpus=num_gpus,
        num_nodes=1,
        strategy="ddp",
        benchmark=True,
        deterministic=False,
        precision=32,
        callbacks=[checkpoint_cb],
        max_epochs=config_yaml["General"]["epochs"],
    )

    # Train!
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)

    # test
    wts = trainer.checkpoint_callback.last_model_path
    trainer.test(model, datamodule=dm, ckpt_path=wts)
