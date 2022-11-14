import yaml

from src.cli.infer_engine import InferEngine

if __name__ == '__main__':
    with open("./configs/config.yaml") as cf:
        config = yaml.safe_load(cf)
    trainer = InferEngine(config=config)
    trainer.run_test()