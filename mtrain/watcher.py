import argparse
import datetime
import json
from collections import OrderedDict
import os

__LABEL__ = ["parameters", "description", "training", "validing"]
__LOGROOT__ = "RECORDS"

if not os.path.exists(__LOGROOT__):
    os.makedirs(__LOGROOT__)


def time_str():
    return datetime.datetime.now().__str__()


class _Watcher:
    def __init__(self):
        note = dict()

        note["description"] = {
            "record_begen_time": time_str(),
        }

        note["parameters"] = dict()
        note["training"] = dict()
        note["validing"] = dict()

        self.note = note
        self.file = None

    def prepare_notes(self, model_name, tag=None):

        model_name = model_name.upper()
        self.note["description"]['model_name'] = model_name
        file_name = (
            model_name
            + "_"
            + datetime.datetime.now().strftime("%m%d_%H%M")
        )
        file_name = file_name + "." + tag if tag is not None else file_name
        file_name += ".json"

        file_path = os.path.join(__LOGROOT__, file_name)
        if os.path.isfile(file_name):
            raise Exception("file exist")
        self.file = open(file_path, "w+")

    def parameter_note(self, params):
        assert type(params) is argparse.Namespace
        self.note["parameters"] = vars(params)

    def loss_note(self, loss_name):
        training_note = self.note["training"]
        if loss_name not in training_note:
            training_note[loss_name] = OrderedDict()
        return training_note[loss_name]

    def to_json(self):
        if self.note and self.file is not None:
            self.note["description"]["record_ending_time"] = time_str()
            json.dump(
                self.note, self.file, indent=4, separators=(",", ": ")
            )
            self.file.close()

watcher = _Watcher()

def parse_watcher_dict(file_name):
    with open(file_name) as watcher_json:
        record_dict = json.load(watcher_json)
        watcher_json.close()
    return record_dict


def parse_losses_record(record_dict):
    losses = record_dict["training"]
    losses_record = {
        name: (losses[name]["step"], losses[name]["record"])
        for name in losses.keys()
    }
    return losses_record



