# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exact Match metric."""
import re
import string

import datasets
import numpy as np

import evaluate

class ExactMatch(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description='',
            citation='',
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            reference_urls=[],
        )

    def _compute(
        self,
        predictions,
        references,
        regexes_to_ignore=None,
        ignore_case=False,
        ignore_punctuation=False,
        ignore_numbers=False,
    ):

        if regexes_to_ignore is not None:
            for s in regexes_to_ignore:
                predictions = np.array([re.sub(s, "", x) for x in predictions])
                references = np.array([re.sub(s, "", x) for x in references])
        else:
            predictions = np.asarray(predictions)
            references = np.asarray(references)

        if ignore_case:
            predictions = np.char.lower(predictions)
            references = np.char.lower(references)

        if ignore_punctuation:
            repl_table = string.punctuation.maketrans("", "", string.punctuation)
            predictions = np.char.translate(predictions, table=repl_table)
            references = np.char.translate(references, table=repl_table)

        if ignore_numbers:
            repl_table = string.digits.maketrans("", "", string.digits)
            predictions = np.char.translate(predictions, table=repl_table)
            references = np.char.translate(references, table=repl_table)

        score_list = predictions == references

        return {"exact_match": np.mean(score_list), "score_list": score_list}
