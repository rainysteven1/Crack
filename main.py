import sys, os, argparse, pathlib, platform, shutil, time
import ujson
from jsonschema import validate

sys.path.insert(0, "./src")

from src import DEVICE, GPU_NAME
from src.core import MODEL_DICT
from src.logger import LoggerFactory, init_file_handler
from src.plot import *
from src.train import train
from src.predict import predict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FOLDER = os.path.join(CURRENT_DIR, "resources/config")
CONFIG_PATH = os.path.join(CONFIG_FOLDER, "configuration.json")
CONFIG_SCHEMA_PATH = os.path.join(CONFIG_FOLDER, "configuration_schema.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Crack")
    model_parser = parser.add_argument_group("Model Options")
    model_parser.add_argument(
        "--state",
        type=str,
        default="train",
        choices=["train", "predict"],
        help="running state",
    )
    model_parser.add_argument(
        "--category",
        type=str,
        default="unet",
        choices=MODEL_DICT.keys(),
        help="running model",
    )
    model_parser.add_argument(
        "--load_model_dir",
        type=str,
        default="",
        help="model directory for loading",
    )
    return parser.parse_args()


def delete_pycache(folder_path):
    for root, dirs, _ in os.walk(folder_path):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)


if __name__ == "__main__":
    args = parse_args()
    with open(CONFIG_PATH, "r") as file:
        config = ujson.load(file)
    with open(CONFIG_SCHEMA_PATH, "r") as file:
        config_schema = ujson.load(file)
    validate(config, config_schema)
    data_path = config["data path"]
    data_attributes = config["data attributes"]
    train_settings = config["train settings"]
    test_settings = config["test settings"]
    result_settings = config["result settings"]

    start_time = time.time()
    str_start_time = time.strftime("%Z-%Y-%m-%d-%H%M%S", time.localtime(start_time))
    running_platform = platform.node()
    file_dir = os.path.abspath(os.path.dirname(__file__))
    if args.state == "train":
        working_dir = pathlib.Path(file_dir).joinpath("result")
        working_dir.mkdir(exist_ok=True)
        working_dir = working_dir.joinpath(str_start_time)
        working_dir.mkdir(exist_ok=True)
        for folder in result_settings["folder_list"]:
            working_dir.joinpath(folder).mkdir(exist_ok=True)
        working_dir = str(working_dir)
        shutil.copy(__file__, f"{working_dir}/code")
        shutil.copytree(f"{file_dir}/src", f"{working_dir}/code/src")
        shutil.copy(CONFIG_PATH, f"{working_dir}/config")
        delete_pycache(f"{working_dir}/code/src")
    elif args.state == "predict" and args.load_model_dir != "":
        working_dir = (
            pathlib.Path(file_dir).joinpath("result").joinpath(args.load_model_dir)
        )

    logger_factory = LoggerFactory()
    file_handler = init_file_handler("%s/log/log_%s.log" % (working_dir, args.state))
    logger_factory.add_file_handler(file_handler)
    logger = logger_factory.logger

    def logger_basic():
        logger.info("Start time: " + str_start_time)
        logger.info("Using {} device".format(DEVICE))
        logger.info("GPU: {}".format(GPU_NAME))
        logger.info("Running platform: " + running_platform)
        logger.info("Running state: " + args.state)
        logger.info("File directory: " + file_dir)
        logger.info("Working directory: " + str(working_dir))
        logger.info(args)

    path_dict = dict()
    for category in data_path["category_list"]:
        path_dict[category] = dict()
        for img_dir in data_path["img_dir_list"]:
            folder = os.path.join(data_path["data_path"], category, img_dir)
            path_dict[category][img_dir] = [
                os.path.join(folder, file_name)
                for file_name in sorted(os.listdir(folder))
            ]

    model_category = args.category

    if args.state == "train":
        load_model_dir = f"{working_dir}/model"
        loss_csv = os.path.join(working_dir, "log", "log_train.csv")
        logger_basic()
        train(
            logger,
            model_category,
            load_model_dir,
            path_dict,
            loss_csv,
            data_attributes,
            train_settings,
        )
        plot_loss(loss_csv, os.path.join(working_dir, "figure", "loss.png"))
        plot_metrics(loss_csv, os.path.join(working_dir, "figure", "metrics.png"))
    elif args.state == "predict":
        best_last = test_settings["best_last"]
        load_model_dir = "%s/model/%s.pth" % (
            working_dir,
            "checkpoint" if best_last == "best" else "model",
        )
        logger_basic()

        data_dict = predict(
            logger,
            model_category,
            load_model_dir,
            path_dict,
            data_attributes,
            test_settings,
        )
        if "data" in data_dict["Prediction"]:
            visualize(
                data_dict,
                list(range(10)),
                os.path.join(working_dir, "figure"),
            )
        else:
            visualize_dict(
                data_dict,
                list(range(10)),
                os.path.join(working_dir, "figure"),
            )
