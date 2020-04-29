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


@DatasetReader.register("trec_6")
class TREC6DatasetReader(DatasetReader):
    """
    Reads tokens and their sentiment labels from the Stanford Sentiment Treebank.
    The Stanford Sentiment Treebank comes with labels
    from 0 to 4. `"5-class"` uses these labels as is. `"3-class"` converts the
    problem into one of identifying whether a sentence is negative, positive, or
    neutral sentiment. In this case, 0 and 1 are grouped as label 0 (negative sentiment),
    2 is converted to label 1 (neutral sentiment) and 3 and 4 are grouped as label 2
    (positive sentiment). `"2-class"` turns it into a binary classification problem
    between positive and negative sentiment. 0 and 1 are grouped as the label 0
    (negative sentiment), 2 (neutral) is discarded, and 3 and 4 are grouped as the label 1
    (positive sentiment).
    Expected format for each input line: a linearized tree, where nodes are labeled
    by their sentiment.
    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        label : `LabelField`
    Registered as a `DatasetReader` with name "sst_tokens".
    # Parameters
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    use_subtrees : `bool`, optional, (default = `False`)
        Whether or not to use sentiment-tagged subtrees.
    granularity : `str`, optional (default = `"5-class"`)
        One of `"5-class"`, `"3-class"`, or `"2-class"`, indicating the number
        of sentiment labels to use.
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
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file.readlines():
                line = line.strip("\n")

                if not line:
                    continue

                words = line.split(" ")
                qclass = words[0].split(":")[0]
                
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
        
        qclass_nums = {
            'ABBR':0,
            'DESC':1,
            'ENTY':2,
            'HUM':3,
            'LOC':4,
            'NYM':5
        }

        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)

        fields: Dict[str, Field] = {"tokens": text_field}
        if qclass is not None:
            fields["label"] = LabelField(qclass_nums[qclass])
        return Instance(fields)