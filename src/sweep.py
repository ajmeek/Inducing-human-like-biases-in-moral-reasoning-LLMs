import wandb
from main import get_config, train

wandb.login()

def sweep():
    sweep_config = dict(
		method = 'random',
		metric = dict(name = 'train_loss', goal = 'minimize'),
		parameters = dict(
            lr=dict(max = 0.1, min = 0.0001, distribution = 'log_uniform_values'),
            batch_size=dict(values = [2, 5, 10, 15, 30]),
            num_epochs=dict(min = 1, max = 4),
		)
	)
    sweep_id = wandb.sweep(sweep=sweep_config, project='AISC_BB')
    wandb.agent(sweep_id=sweep_id, function=objective, count=30)

def objective():
    wandb.init(project='AISC_BB')
    config = get_config()
    config['shuffle_train']=True
    config['regularization_coef']=0.1
    config['regularize_from_init']=False
    config['batches_per_epoch']=100
    config['num_samples_train']=300
    for k in wandb.config.keys():
        config[k] = wandb.config.get(k)
    train(config)

if __name__ == '__main__':
    sweep()