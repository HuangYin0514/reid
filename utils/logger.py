import logging
import os


class Logger(logging.Logger):
    """Inherit from logging.Logger.
    Print logs to console and file.
    """

    def __init__(self, dir_path):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        super(Logger, self).__init__("Training logger")

        # Print logs to console and file
        file_handler = logging.FileHandler(os.path.join(self.dir_path, "train_log.txt"))
        console_handler = logging.StreamHandler()
        log_format = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        self.addHandler(file_handler)
        self.addHandler(console_handler)

