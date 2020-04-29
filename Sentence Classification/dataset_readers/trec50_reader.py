from typing import Dict, List
import logging

from overrides import overrides
from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


@DatasetReader.register("trec_50")
class TREC50DatasetReader(DatasetReader):
    """
    TREC50 dataset reader
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
         with open(cached_path(file_path), "r",encoding = "ISO-8859-1") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file.readlines():
                line = line.strip("\n")

                if not line:
                    continue

                words = line.split(" ")
                qclass = words[0]
                
                instance = self.text_to_instance(words[1:], qclass)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(
        self, tokens: List[str], qclass: str = None
    ) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        # Parameters
        tokens : `List[str]`, required.
            The tokens in a given sentence.
        sentiment : `str`, optional, (default = None).
            The sentiment for this sentence.
        # Returns
        An `Instance` containing the following fields:
            tokens : `TextField`
                The tokens in the sentence or phrase.
            label : `LabelField`
                The sentiment label of the sentence or phrase.
        """
           #   qclass_nums = {
        #     'ABBR':0,
        #     'DESC':1,
        #     'ENTY':2,
        #     'HUM':3,
        #     'LOC':4,
        #     'NYM':5
        # } 
        qclass_set = set(qclass)
        print(qclass_set)
        qclass_num = {}
        k = 0
        for i in qclass_set:
          qclass_num[i] = k
          k+=1

        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)

        fields: Dict[str, Field] = {"tokens": text_field}
        if qclass is not None:
            fields["label"] = LabelField(qclass_nums[qclass])
        return Instance(fields)