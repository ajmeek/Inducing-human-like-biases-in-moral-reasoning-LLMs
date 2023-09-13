import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
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
    if context.early_stop_threshold is not None:
        # Stop when no val accuracy improvement after 100 epochs.
        callbacks.append(
            EarlyStopping(
                PLModel.VAL_ACC,
                mode="max",
                verbose=True,
                stopping_threshold=context.early_stop_threshold,
            )
        )
    if context.pltc.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                monitor=PLModel.VAL_ACC,
                dirpath=Context.artifactspath,
                verbose=True,
                mode="max",
                auto_insert_metric_name=True,
                every_n_epochs=3,
                save_top_k=1,
            )
        )
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=context.artifactspath,
        callbacks=callbacks,
        **vars(context.pltc)
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
