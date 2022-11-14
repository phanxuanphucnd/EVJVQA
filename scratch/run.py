import yaml

from src.cli.train_engine import TrainEngine

if __name__ == '__main__':
    with open("./configs/config.yaml") as cf:
        config = yaml.safe_load(cf)
    trainer = TrainEngine(config=config)
    trainer.train()