import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime
from json import loads as jsloads
from pprint import pprint
from simple_parsing import ArgumentParser
import wandb
from transformers import AutoModel, AutoTokenizer
from Context import Context
from Context import Context
from model import BERT
from pl_model import LitBert
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
    model = BERT(base_model, data_module)
    lit_model = LitBert(model, context.plc, data_module)

    logger = WandbLogger(save_dir=context.artifactspath, project="AISC_BB")
    logger.log_hyperparams(jsloads(context.dumps_json()))

    # train the model
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=logger,
        default_root_dir=context.artifactspath,
        **vars(context.pltc),
    )
    print("Fitting...")
    trainer.fit(lit_model, data_module)

    print("Testing...")
    trainer.test(lit_model, data_module)

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
