import wandb
from main import get_config
from main import train
#from utils.constants import Sampling

wandb.login()

def sweep():
    sweep_config = dict(
		method = 'random',
		metric = dict(name = 'train_loss', goal = 'minimize'),
		parameters = dict(
            lr = dict(max = 0.1, min = 0.0001, distribution = 'log_uniform_values'),
            batch_size = dict(values = [2, 5, 7, 10]),
            #sampling_method = dict(values = ['LAST', 'SENTENCES', 'AVG'])
		)
	)
    sweep_id = wandb.sweep(sweep=sweep_config, project='AISC_BB')
    wandb.agent(sweep_id=sweep_id, function=objective, count=30)

def objective():
    wandb.init(project='AISC_BB')
    config = get_config()
    config['checkpoint']='deepset/deberta-v3-large-squad2'
    config['shuffle_train']=True
    config['regularization_coef']=0.1
    config['regularize_from_init']=False
    config['num_epochs'] = 5
    config['only_train_head'] = False
    #config['sampling_method'] = Sampling.SENTENCES
    for k in wandb.config.keys():
        config[k] = wandb.config.get(k)
    train(config)

if __name__ == '__main__':
    raise NotImplemented()
    sweep()