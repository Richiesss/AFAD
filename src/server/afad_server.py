import flwr as fl

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("AFADServer")


class AFADServer:
    def __init__(self, config: dict):
        self.config = config
        self.num_rounds = config["experiment"]["num_rounds"]

    def start(self):
        logger.info("Starting AFAD Server...")
        # Strategy initialization (TODO)
        strategy = fl.server.strategy.FedAvg()  # Placeholder

        fl.server.start_server(
            server_address=self.config["server"]["address"],
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )


if __name__ == "__main__":
    # Test run
    try:
        config = load_config("config/afad_config.yaml")
        server = AFADServer(config)
        # server.start() # Commented out to prevent actual start during import
    except Exception as e:
        logger.error(f"Server init failed: {e}")
