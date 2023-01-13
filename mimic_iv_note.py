"""Huggingface dataset loader for MIMIC-IV-NOTE dataset."""


import csv
import gzip
import io
import os

import datasets


_CITATION = """\
@misc{mimic-iv-note_v2p2,
  doi = {10.13026/1N74-NE17},
  url = {https://physionet.org/content/mimic-iv-note/2.2/},
  author = {Johnson, Alistair and Pollard, Tom and Horng, Steven and Celi, Leo Anthony and Mark, Roger},
  title = {MIMIC-IV-Note: Deidentified free-text clinical notes},
  publisher = {PhysioNet},
  year = {2023}
}
"""

_DESCRIPTION = """\
The advent of large, open access text databases has driven advances in state-of-the-art model performance in natural language processing (NLP). The relatively limited amount of clinical data available for NLP has been cited as a significant barrier to the field's progress. Here we describe MIMIC-IV-Note: a collection of deidentified free-text clinical notes for patients included in the MIMIC-IV clinical database. MIMIC-IV-Note contains 331,794 deidentified discharge summaries from 145,915 patients admitted to the hospital and emergency department at the Beth Israel Deaconess Medical Center in Boston, MA, USA. The database also contains 2,321,355 deidentified radiology reports for 237,427 patients. All notes have had protected health information removed in accordance with the Health Insurance Portability and Accountability Act (HIPAA) Safe Harbor provision. All notes are linkable to MIMIC-IV providing important context to the clinical data therein. The database is intended to stimulate research in clinical natural language processing and associated areas.
"""

_HOMEPAGE = "https://physionet.org/content/mimic-iv-note/2.2/"

_LICENSE = "https://physionet.org/content/mimic-iv-note/view-license/2.2/"



class MimicIVNote(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("2.2.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="all",
            version=VERSION,
            description="Discharge summaries and radiology reports"
        ),
        datasets.BuilderConfig(
            name="discharge",
            version=VERSION,
            description="Discharge summaries only"
        ),
        datasets.BuilderConfig(
            name="radiology",
            version=VERSION,
            description="Radiology reports only"
        ),
    ]

    DEFAULT_CONFIG_NAME = "all"

    def _info(self):

        features = datasets.Features(
            {
                "note_id": datasets.Value("string"),
                "subject_id": datasets.Value("string"),
                "hadm_id": datasets.Value("string"),
                "note_type": datasets.Value("string"),
                "note_seq": datasets.Value("string"),
                "charttime": datasets.Value("string"),
                "storetime": datasets.Value("string"),
                "text": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        if self.config.data_dir is None:
            raise ValueError(
                "This loader requires a local path. "
                "Please pass the data_dir kwarg to load_dataset."
            )
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": str(datasets.Split.TRAIN),
                },
            ),
        ]

    def _generate_examples(self, data_dir, split):

        _id = 0

        if self.config.name in ["discharge", "all"]:
            with gzip.GzipFile(os.path.join(data_dir, "discharge.csv.gz")) as gzf:
                reader = csv.DictReader(io.TextIOWrapper(gzf))
                for row in reader:
                    yield _id, row
                    _id += 1

        if self.config.name in ["radiology", "all"]:
            with gzip.GzipFile(os.path.join(data_dir, "radiology.csv.gz")) as gzf:
                reader = csv.DictReader(io.TextIOWrapper(gzf))
                for row in reader:
                    yield _id, row
                    _id += 1
