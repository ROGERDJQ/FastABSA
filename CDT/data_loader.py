'''
Author: Roger
Date: 2020-12-01 00:14:02
LastEditors  : Roger
LastEditTime : 2021-02-05 10:42:49
Description: data_loader. Specifically, ABSALoader can convert formatted file into databundle in fastNLP. 
            CDTPipe is designed for CDT reproducation.
'''
from fastNLP import Const, DataSet
from fastNLP.io import DataBundle, Loader, Pipe
from fastNLP.io.pipe.utils import _add_words_field, _indexize
from fastNLP.io.utils import check_loader_paths
import json

class CDTLoader(Loader):
    '''
    description: Load absa data from data file. 
                Note that the data file is already preprocessed from original raw data,
                which would be released in the future. 
    '''
    def __init__(self):
        super().__init__()

    def _load(self, path, is_lower=True):
        # 
        with open(path, "r", encoding="utf8") as f:
            raw_data = json.load(f)
        data = []
        for d in raw_data:
            tok = d["token"]
            pos = d["pos"]
            head = d["head"] if not is_lower else [
                w.lower() for w in d["token"]
            ]
            deprel = d["deprel"]
            for aspect in d["aspects"]:
                asp = [aspect["term"], ["unk"]][len(aspect["term"]) == 0]
                fidx = aspect["from"]
                tidx = aspect["to"]
                pol = aspect["polarity"]
                data.append([tok, pos, head, deprel, asp, fidx, tidx, pol])
        fields = [
            Const.RAW_WORD, "pos", "dephead", "deprel", "asp", "fidx", "tidx",
            Const.TARGET
        ]
        ds = DataSet(dict(zip(fields, zip(*data))))
        return ds

    def load(self, paths, is_lower):
        if paths is None:
            raise NotImplementedError(f"I am not ready for downloading!")
        paths = check_loader_paths(paths)
        datasets = {
            name: self._load(path, is_lower)
            for name, path in paths.items()
        }
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class CDTPipe(Pipe):
    def __init__(self, is_lower: bool = False):
        self.is_lower = is_lower


    def _get_post(self,instance):
        fidx = instance["fidx"]
        tidx = instance["tidx"]
        seq_len = instance["seq_len"]
        return ([i - fidx
                    for i in range(fidx)] + [0 for _ in range(fidx, tidx)] +
                [i - tidx + 1 for i in range(tidx, seq_len)])

    def _get_mask(self,instance):
        if not instance["fidx"] == instance["tidx"]:
            return [[0, 1][i in range(instance["fidx"], instance["tidx"])]
                    for i in range(instance["seq_len"])]
        else:
            return [1 for _ in range(instance["seq_len"])]

    def process(self, data_bundle):
        data_bundle = _add_words_field(data_bundle)
        data_bundle = _indexize(
            data_bundle=data_bundle,
            input_field_names=[Const.INPUT, "pos", "deprel"])
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
            data_bundle.get_vocab(Const.INPUT).index_dataset(dataset,field_name="asp",new_field_name="aspect")
        data_bundle.apply(self._get_post, new_field_name="post")
        data_bundle.apply(self._get_mask, new_field_name="aspmask")
        data_bundle.set_input(Const.INPUT, Const.INPUT_LEN, "pos", "dephead",
                              "deprel", "aspect", "fidx", "tidx", "post","aspmask")
        data_bundle.set_target(Const.TARGET)
        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = CDTLoader().load(paths, is_lower=self.is_lower)
        return self.process(data_bundle)
