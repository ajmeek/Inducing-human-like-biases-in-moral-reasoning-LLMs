from Context import Context
from datetime import datetime
from json import loads as jsloads
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner import Tuner
from pl_model import PLModel
from pprint import pprint
from simple_parsing import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from utils.BrainBiasDataModule import BrainBiasDataModule
import lightning.pytorch as pl
import wandb
import torch


def train(context: Context):
    pprint(context)
    context.artifactspath.mkdir(parents=True, exist_ok=True)

    base_model = AutoModel.from_pretrained(context.model_path)
    tokenizer = AutoTokenizer.from_pretrained(context.model_path)

    # use_ia3_layers = False
    # if use_ia3_layers:
    #     from ia3_model_modifier import modify_with_ia3
    #     layers_to_replace_with_ia3 = "key|value|intermediate.dense"
    #     base_model = modify_with_ia3(base_model, layers_to_replace_with_ia3)

    data_module = BrainBiasDataModule(
        context.get_ds_configs(), tokenizer, num_workers=context.num_workers
    )
    model = PLModel(context.plc, data_module, base_model)
    if context.checkpoint_path:
        print(f"Loading weights only from {context.checkpoint_path}.")
        ckpt = torch.load(context.checkpoint_path)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    logger = WandbLogger(save_dir=context.artifactspath, project="AISC_BB")
    logger.log_hyperparams(jsloads(context.dumps_json()))

    # train the model
    callbacks = []
    if context.early_stop_threshold is not None:
        # Stop when no val accuracy improvement after 100 epochs.
        callbacks.append(
            EarlyStopping(
                model.main_val_metric_name,
                mode="max",
                verbose=True,
                stopping_threshold=context.early_stop_threshold,
            )
        )
    if context.pltc.enable_checkpointing:
        fn = (
            logger.experiment.name or ""
        ) + "{epoch}-{val_acc:.2f}-{train_loss:.2f}-{step}"
        callbacks.append(
            ModelCheckpoint(
                monitor=model.main_val_metric_name,
                dirpath=Context.artifactspath,
                filename=fn,
                verbose=True,
                mode="max",
                auto_insert_metric_name=True,
                every_n_epochs=3,
                save_top_k=1,
            )
        )
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    profiler = (
        PyTorchProfiler(filename="profiling-results", export_to_chrome=True)
        if context.profiler == "pytorch"
        else context.profiler
    )
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=context.artifactspath,
        callbacks=callbacks,
        profiler=profiler,
        **vars(context.pltc),
    )
    if context.find_learning_rate:
        print("Finding learning rate...")
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, data_module)
        print(lr_finder.results)
        suggested_lr = lr_finder.suggestion()
        context.plc.adamw.lr = suggested_lr
        model.learning_rate = suggested_lr

    print("Fitting...")
    trainer.fit(model, data_module)

    print("Testing...")
    # Warning. This uses the best weights (not always last):
    # See https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-loop
    # Also don't run on parallel setting because possible a same batch used.
    trainer.test(model, data_module)

    logger.save()
    wandb.finish()
    if context.last_checkpoint_path:
        print(f"Saving checkpoint to {context.last_checkpoint_path}")
        trainer.save_checkpoint(context.last_checkpoint_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Context(), dest="context")
    args = parser.parse_args()
    context = args.context
    train(context)
