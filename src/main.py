import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime
from json import loads as jsloads
from pprint import pprint
from simple_parsing import ArgumentParser
import wandb
from transformers import AutoModel, AutoTokenizer
from Context import Context
from Context import Context
from pl_model import PLModel
from utils.BrainBiasDataModule import BrainBiasDataModule


def train(context: Context):
    pprint(context)

    base_model = AutoModel.from_pretrained(context.model_path)
    tokenizer = AutoTokenizer.from_pretrained(context.model_path)

    # use_ia3_layers = False
    # if use_ia3_layers:
    #     from ia3_model_modifier import modify_with_ia3
    #     layers_to_replace_with_ia3 = "key|value|intermediate.dense"
    #     base_model = modify_with_ia3(base_model, layers_to_replace_with_ia3)

    data_module = BrainBiasDataModule(context.get_ds_configs(), tokenizer)
    model = PLModel(base_model, context.plc, data_module)

    logger = WandbLogger(save_dir=context.artifactspath, project="AISC_BB")
    logger.log_hyperparams(jsloads(context.dumps_json()))

    # train the model
    callbacks = []
    if context.is_early_stop:
        # Stop when no val accuracy improvement after 100 epochs.
        callbacks.append(EarlyStopping(
            PLModel.VAL_ACC,
            mode='max', 
            stopping_threshold=0.8,  # Stop if >=0.8
            min_delta=0.1, # Expect more than 0.1 improvement.
            patience=100,
            check_on_train_epoch_end=True
        ))
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=logger,
        default_root_dir=context.artifactspath,
        callbacks=callbacks,
        **vars(context.pltc),
    )
    print("Fitting...")
    trainer.fit(model, data_module)

    print("Testing...")
    # Warning. This uses the best weights (not always last):
    # See https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-loop
    # Also don't run on parallel setting because possible a same batch used.
    trainer.test(model, data_module)

    logger.save()
    wandb.finish()
    if context.to_save_model:
        trainer.save_checkpoint(
            context.artifactspath
            / f'model-{datetime.utcnow().isoformat(timespec="minutes").replace(":","")}.ckpt'
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Context(), dest="context")
    args = parser.parse_args()
    context = args.context
    train(context)
