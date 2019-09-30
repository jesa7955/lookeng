import logging
import sys
import os
import json

import pandas as pd
import numpy as np
import gokart
import luigi
import jaconv
from pyknp import Juman


class ChabsaJsonReader(gokart.TaskOnKart):
    task_namespace = "m3lookeng"
    base_path = luigi.Parameter()

    def requires(self):
        pass

    def output(self):
        return self.make_target("chaABSA_sentihood_format.json")

    def run(self):
        results = []
        index = 0
        directory = os.path.abspath(self.base_path)
        for _, _, file_names in os.walk(directory):
            for file_name in file_names:
                file_path = os.fsdecode(os.path.join(directory, file_name))
                if file_path.endswith(".json"):
                    with open(file_path) as file_obj:
                        for sentence in json.load(file_obj)['sentences']:
                            sentence_id, index = index, index + 1
                            text = sentence['sentence']
                            opinions = [{
                                'sentiment':
                                opinion['polarity'],
                                'aspect':
                                opinion['category'].split('#')[1],
                                'target':
                                opinion['target']
                            } for opinion in sentence['opinions']]
                            results.append({
                                'opinions': opinions,
                                'id': sentence_id,
                                'text': text,
                            })
        self.dump(pd.DataFrame(results))


class GenerateTextTargetPairs(gokart.TaskOnKart):
    task_namespace = "m3lookeng"
    absa_base_path = luigi.Parameter()
    task_name = luigi.Parameter()
    logger = logging.getLogger("luigi")

    def requires(self):
        return ChabsaJsonReader(base_path=self.absa_base_path)

    def output(self):
        return self.make_target("chaABSA_sentence_target_pair.csv")

    def run(self):
        source = self.load()
        results = []
        for _, sentence in source.iterrows():
            for opinion in sentence['opinions']:
                results.append({
                    'sentence': sentence['text'],
                    'target': opinion['target'],
                    'aspect': opinion['aspect'],
                    'sentiment': opinion['sentiment']
                })
        self.dump(pd.DataFrame(results))


class GenearteSentimentAnalysisData(gokart.TaskOnKart):
    task_namespace = "m3lookeng"
    task_name = luigi.Parameter()
    absa_base_path = luigi.Parameter()

    def requires(self):
        return GenerateTextTargetPairs(absa_base_path=self.absa_base_path,
                                       task_name=self.task_name)

    def output(self):
        return self.make_target(f"{self.task_name}.tsv")

    def run(self):
        data = self.load()
        jumanpp = Juman()
        output = []
        for _, row in data.iterrows():
            zenkaku = jaconv.h2z(row["sentence"], ascii=True, digit=True)
            splited = [
                mrph.midasi for mrph in jumanpp.analysis(zenkaku).mrph_list()
            ]
            if self.task_name == 'QA_B':
                qa_zenkaku = jaconv.h2z(
                    f"{row['target']}の{row['aspect']}は{row['sentiment']}",
                    ascii=True,
                    digit=True,
                )
            else:
                qa_zenkaku = "　"
            qa_splited = [
                mrph.midasi
                for mrph in jumanpp.analysis(qa_zenkaku).mrph_list()
            ]
            output.append({
                "context": " ".join(splited),
                "qa": " ".join(qa_splited),
                "label": 1
            })
        self.dump(pd.DataFrame(output))


if __name__ == "__main__":
    luigi.configuration.LuigiConfigParser.add_config_path('./conf/param.ini')
    np.random.seed(57)
    gokart.run()
