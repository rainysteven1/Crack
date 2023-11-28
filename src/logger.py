import logging


class LoggerFactory:
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def add_file_handler(self, file_handler):
        self.logger.addHandler(file_handler)

    def change_file_handler(self, old_handler, new_handler):
        self.logger.removeHandler(old_handler)
        self.logger.addHandler(new_handler)


def init_file_handler(filename):
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    return file_handler
