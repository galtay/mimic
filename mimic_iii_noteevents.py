"""Huggingface dataset loader for MIMIC-III NOTEEVENTS dataset."""


import csv
import gzip
import io
import os

import datasets


_CITATION = """\
@misc{mimiciii-noteevents_v1p4,
  doi = {10.13026/C2XW26},
  url = {https://physionet.org/content/mimiciii/1.4/},
  author = {Johnson, Alistair and Pollard, Tom and Mark, Roger},
  title = {MIMIC-III Clinical Database},
  publisher = {PhysioNet},
  year = {2016}
}
"""

_DESCRIPTION = """\
MIMIC-III is a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012. The database includes information such as demographics, vital sign measurements made at the bedside (~1 data point per hour), laboratory test results, procedures, medications, caregiver notes, imaging reports, and mortality (including post-hospital discharge).

MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical decision-rule improvement, and electronic tool development. It is notable for three factors: it is freely available to researchers worldwide; it encompasses a diverse and very large population of ICU patients; and it contains highly granular data, including vital signs, laboratory results, and medications.
"""

_HOMEPAGE = "https://physionet.org/content/mimiciii/1.4/"

_LICENSE = "https://physionet.org/content/mimiciii/view-license/1.4/"



class MimicIIINoteEvents(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.4.0")

    def _info(self):

        features = datasets.Features(
            {
                "row_id": datasets.Value("string"),
                "subject_id": datasets.Value("string"),
                "hadm_id": datasets.Value("string"),
                "chartdate": datasets.Value("string"),
                "charttime": datasets.Value("string"),
                "storetime": datasets.Value("string"),
                "category": datasets.Value("string"),
                "description": datasets.Value("string"),
                "cgid": datasets.Value("string"),
                "iserror": datasets.Value("string"),
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

        with gzip.GzipFile(os.path.join(data_dir, "NOTEEVENTS.csv.gz")) as gzf:
            reader = csv.DictReader(io.TextIOWrapper(gzf))
            for row in reader:
                yield _id, {k.lower():v for k,v in row.items()}
                _id += 1
