# https://huggingface.co/datasets/bdsaglam/musique/blob/main/musique.py
# with bugfixes

import json
from itertools import chain
import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """[MuSiQue](https://arxiv.org/pdf/2108.00573.pdf)"""
_NAME = "musique"
_VERSION = "1.0.0"
_CITATION = """
@article{trivedi2021musique,
  title={{M}u{S}i{Q}ue: Multihop Questions via Single-hop Question Composition},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022}
  publisher={MIT Press}
}
"""

_HOME_PAGE = "https://github.com/StonyBrookNLP/musique"
_DATA_URL = f'https://huggingface.co/datasets/bdsaglam/{_NAME}/resolve/main/data'
_URL_MAP = {
    'answerable': {
        str(datasets.Split.TRAIN): f'{_DATA_URL}/musique_ans_v1.0_train.jsonl',
        str(datasets.Split.VALIDATION): f'{_DATA_URL}/musique_ans_v1.0_dev.jsonl',
        str(datasets.Split.TEST): f'{_DATA_URL}/musique_ans_v1.0_test.jsonl',
    },
    'full': {
        str(datasets.Split.TRAIN): f'{_DATA_URL}/musique_full_v1.0_train.jsonl',
        str(datasets.Split.VALIDATION): f'{_DATA_URL}/musique_full_v1.0_dev.jsonl',
        str(datasets.Split.TEST): f'{_DATA_URL}/musique_full_v1.0_test.jsonl',
    },
}


class MusiqueConfig(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(self, **kwargs):
        """BuilderConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class Musique(datasets.GeneratorBasedBuilder):
    """Dataset."""

    BUILDER_CONFIGS = [
        MusiqueConfig(name='answerable', version=datasets.Version(_VERSION), description=f"{_DESCRIPTION}-Answerable"),
        MusiqueConfig(name='full', version=datasets.Version(_VERSION), description=f"{_DESCRIPTION}-Full"),
    ]

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URL_MAP[self.config.name])
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files[str(datasets.Split.TRAIN)]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files[str(datasets.Split.VALIDATION)]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files[str(datasets.Split.TEST)]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            from collections import defaultdict
            count = defaultdict(int)
            for line in f:
                record = json.loads(line)
                #record.pop('answer_aliases', None)
                #yield record['id'], record
                count[record['id']] += 1
                assert count[record['id']] <= 2
                key = (record['id'], count[record['id']])
                yield str(key), record

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "paragraphs": datasets.features.Sequence(
                        {
                            "idx": datasets.Value("int32"),
                            "title": datasets.Value("string"),
                            "paragraph_text": datasets.Value("string"),
                            "is_supporting": datasets.Value("bool"),
                        }
                    ),
                    "question": datasets.Value("string"),
                    "question_decomposition": datasets.features.Sequence(
                        {
                            "id": datasets.Value("int32"),
                            "question": datasets.Value("string"),
                            "answer": datasets.Value("string"),
                            "paragraph_support_idx": datasets.Value("int32"),
                        }
                    ),
                    "answer": datasets.Value("string"),
                    "answer_aliases": datasets.features.Sequence(datasets.Value("string")),
                    "answerable": datasets.Value("bool"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage=_HOME_PAGE,
            citation=_CITATION,
            # task_templates=[
            #     QuestionAnsweringExtractive(
            #         question_column="question", context_column="context", answers_column="answers"
            #     )
            # ],
        )
