import sys
import logging
import datetime
import os
from mtrain.train_capsule import LossBuket
import re
import colorlog
# from mmodel.basic_params import get_param_parser
from mtrain.watcher import watcher


BUILD = 50
TRAIN = 51
VALID = 52
WARMN = 53
HINTS = 54
STORE = 60

logging.addLevelName(BUILD, "BUILD")
logging.addLevelName(TRAIN, "TRAIN")
logging.addLevelName(VALID, "VALID")
logging.addLevelName(WARMN, "WARMN")
logging.addLevelName(STORE, "STORE")
logging.addLevelName(HINTS, "HINTS")

LOG_DIR = ""



class GLOBAL(object):
    _TAG_ = None
    _CHECK_POINT_FLODER_ = "./_CKTP/"
    _LOGER_FLODER_ = "./_MLOGS/"
    _SUMMARY_LOG_DIR = "./SUMMARY.log"


def path_helper(base_folder_name, tag_name, create=True):
    """get a path acordding `base_folder_name` and `tat_name`
    """

    folder_name = base_folder_name

    # add tag name to path
    if tag_name is not None:
        for i in tag_name.split("/"):
            if len(i) != 0:
                folder_name += i + "/"

    # add time stamp to folder
    time_folder_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    folder_name += time_folder_name + "/"

    if create:
        if not os.path.exists((folder_name)):
            os.makedirs(folder_name)

    return folder_name


def setup_log_folder_name(
    tag_name=None, base_folder_name=GLOBAL._LOGER_FLODER_, create=True
):
    """Set global path to store log file
    
    Keyword Arguments:
        base_folder_name {string} -- the dir for base folder. (default: {None})
        create {bool} -- create if folder not exist. (default: {True})
    """

    global LOG_DIR
    LOG_DIR = path_helper(
        base_folder_name=base_folder_name, tag_name=param.tag
    )


def read_step_and_loss(**kwargs):
    result = dict()
    pattern = re.compile(r"\[(.*?)\]")
    for _, name in enumerate(kwargs):
        with open(kwargs[name], mode="r") as f:
            xs = list()
            ys = list()
            for line in f.readlines():
                (x, y) = pattern.findall(line)
                xs.append(float(x))
                ys.append(float(y))
            result[name] = (xs, ys)
    return result


def get_colored_logger(name):

    COLOR_FORMATTER = colorlog.ColoredFormatter(
        "%(log_color)s %(levelname)s %(reset)s  %(bold_white)s⯈   %(message_log_color) s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "BUILD": "fg_black,bg_blue",
            "TRAIN": "fg_black,bg_yellow",
            "VALID": "fg_black,bg_green",
            "WARMN": "fg_black,bg_red",
            "HINTS": "fg_black,bg_cyan",
        },
        secondary_log_colors={
            "message": {
                "BUILD": "blue",
                "TRAIN": "yellow",
                "VALID": "green",
                "WARMN": "red",
                "HINTS": "cyan",
            }
        },
        style="%",
    )

    handler = colorlog.StreamHandler()
    handler.setFormatter(COLOR_FORMATTER)

    logger = colorlog.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

from collections import OrderedDict

class LogCapsule(object):
    def __init__(
        self,
        loss_bucker: LossBuket,
        name,
        step=1,
        to_file=False
    ):

        self.tag = name
        logger = logging.getLogger(name)

        self.loss_record = None
        if to_file:
            self.note = watcher.loss_note(self.tag)
            self.note['step'] = step
            self.note['record'] = list()
            self.loss_record = self.note['record']
        

        self.LOSS_FORMAT = ">%-12s< at step [%6d] -> [%.3f]."
        self.PRT_FORMAT = "At >%-12s< stage, loss is [%.3f]."

        self.logger = logger
        self.range_loss = None
        self.range_step = 0
        self.loss_bucket = loss_bucker

    def update_record(self, v):
        self.loss_bucket.value = v
        self.record()

    def record(self):
        # closs = self.loss_bucket.value.item()
        try:
            closs = self.loss_bucket.value.clone().detach()
        except:
            closs = self.loss_bucket.value
        if self.range_loss is None:
            self.range_loss = closs
        else:
            self.range_loss += closs
        
        self.range_step += 1.0

    def log_current_avg_loss(self, step=None):

        try:
            result = self.range_loss / self.range_step
        except:
            result = 0.0
        self.range_loss = None
        self.range_step = 0.0

        try:
            loss = result.item()
        except:
            loss = result

        if self.loss_record is not None:
            self.loss_record.append(loss)

        return loss

    def avg_record(self):
        assert False
        try:
            result = self.range_loss / (self.range_step * 1.0)
        except:
            result = 0.0
        self.range_loss = None
        self.range_step = 0.0
        return result.item()

    def _log_to_same_loger(self, s):
        self.logger.log(self.mode, s)
        return


if __name__ == "__main__":

    """testing
    """
    read_step_and_loss(
        haha="G:\\VS Code\\Unshifting-Mask\\_MLOGS\\18-11-29-20-11\\cross_entropy.log"
    )
    # setup_log_folder_name()
    # log = LLoger("ADAM")
    # log.loss(5, 5.0)

