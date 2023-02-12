from dataclasses import dataclass, field
import json
from multiprocessing import context
from typing import Any, Optional, Dict, List, Union

try:
    from typing import Literal 
except ImportError: 
    from typing_extensions import Literal

import pandas as pd
import numpy as np

from dataclasses import asdict

from pydantic import BaseConfig
from pydantic.json import pydantic_encoder
@dataclass
class Document: 
    content: Union[str, pd.DataFrame]
    content_type: Literal["text", "table"]
    id: str
    meta: Dict[str, Any]
    score: Optional[float] = None
    embedding: Optional[np.ndarray] = None

    def __init__(
        self, 
        content: Union[str, pd.DataFrame], 
        content_type: Literal["text", "table"] = "text", 
        id: Optional[str] = None, 
        score: Optional[float] = None,
        meta: Dict[str, Any] = None, 
        embedding: Optional[np.ndarray] = None,  
        ):
        """

        """
        if content is None: 
            raise ValueError(f"Can't create 'Document': Mandatory 'content' field is None")
        
        self.content = content
        self.content_type = content_type, 
        self.score = score
        self.meta = meta or {}

        if embedding is not None: 
            embedding = np.asarray(embedding)
        self.embedding = embedding

        self.id = id           

    def to_dict(self, field_map = {}) -> Dict: 
        """
        Convert Document to dict. An optional field_map can be supplied to change the names of the keys in the 
        resulting dict. This way you can work with standardized Document objects, but adjust the format that 
        they are serialized/ stored in other places 
        Example: 
        | doc = Document(content="some text", content_type="text")
        | doc.to_dict(field_map = {"custom_current_field": "content"})
        | >>>{"custom_current_field": "some_text", "content_type":text}

        """
        inv_field_map = {v: k for k, v in field_map.items()}
        _doc: Dict[str, str] = {}
        for k, v in self.__dict__.items(): 
            if k == "content": 
                #convert pd.DataFrame to list of rows for serialization 
                if self.content_type == "table" and isinstance(self.content, pd.DataFrame): 
                    v = [self.content.columns.tolist()] + self.content.values.tolist()
            k = k if k not in inv_field_map else inv_field_map[k]
            _doc[k] = v
        return _doc
    
    def to_json(self, field_map = {}) -> str: 
        d = self.to_dict(field_map=field_map)
        j = json.dumps(d, cls=NumpyEncoder)

    @classmethod
    def from_dict(
        cls, dict: Dict[str, Any], field_map: Dict[str, Any] = {}
    ): 
        _doc = dict.copy()
        init_args = ["content", "content_type", "id", "score", "question", "meta", "embedding"]
        if "meta" not in _doc.keys(): 
            _doc["meta"] = {}
        #copy additional fields into "meta"
        for k, v in _doc.items(): 
            if k not in init_args and k not in field_map: 
                _doc["meta"][k] = v
        #remove additional fields from top level 
        _new_doc = {}
        for k, v in _doc.items(): 
            if k in init_args: 
                _new_doc[k] = v
            elif k in field_map: 
                k = field_map[k]
                _new_doc[k] = v

        #convert list of rows to pd.Dataframe
        if _new_doc.get("content_type", None) == "table" and isinstance(_new_doc["content"], list): 
            _new_doc["content"] = pd.DataFrame(columns=_new_doc["content"][0], data=_new_doc["content"][1:])

        return cls(**_new_doc)

    @classmethod
    def from_json(cls, data: str, field_map = {}): 
        d = json.loads(data)
        return cls.from_dict(d, field_map=field_map)

    def __repr__(self):
        return f"<Document: {str(self.to_dict())}>"
    
    def __str__(self):
        #In some cases, self.content is None (therefore not subcriptable)
        if self.content is None: 
            if self.id:
                return f"<Document: id= {self.id}, content=None>"
            else:
                return f"<Document: content=None>"
        if self.id: 
            return f"<Document: id={self.id}, content='{self.content[:100]} {'...' if len(self.content) > 100 else ''}'>"
        return f"<Document: content='{self.content[:100]} {'...' if len(self.content) > 100 else ''}'>"

    def __lt__(self, other): 
        """Enable sorting of Documents by score"""
        return self.score < other.score

@dataclass
class Span: 
    start: int 
    end: int
    """
    Defining a sequence of characters (Text span) or cells (Table span) via start and end index. 
    For extractive QA: Character where answer starts/ends
    For TableQA: Cell where the answer starts/ends (counted from top left to bottom right of table)

    :param 
        start: Position where the span starts 
        end: Position where the span ends 
    """

@dataclass
class Answer: 
    answer: str 
    type: Literal["generative, extractive, other"] = "generative"
    score: Optional[float] = None
    context: Optional[Union[str, pd.DataFrame]] = None
    offsets_in_document: Optional[List[Span]] = None
    offsets_in_context: Optional[List[Span]] = None
    document_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    embedding = None

    """
    The fundamental object to represent any type of Answers (e.g. generative QA, ...)
    For example, it's used within some Nodes like the Generator, but also in the REST API

    :param 
        answer: The answer string. If There's no possible answer (aka "no_answer" or "is_impossible") this will be an empty string.
        type:   One of ("generative", "extractive", "other"): whether this answer comes from an extractive model
                (i.e. we can locate an exact answer string in one of the documents) or from a generative model
                (i.e. no pointer to a specific document, no offsets)
        score:  The relevance score of the Answer determined by a model (Reader or Generator)   . 
                In the range of [0,1], where 1 means extremely relevant
        context:    the related content that was used to create the answer (i.e. a text passage, part of a table, image...)
        offsets_in_document:    List of `Span` objects with start and end positions of the answer in the 
                                docuument** (as stored in the document store). 
                                For example, ExtractiveQA: Character where answer starts => `Answer.offsets_in_document[0].start
                                For TableQA: Cell where the answer starts (counted from top left to bottom right of table) => `Answer.offsets_in_document[0].start
                                (Note that in TableQA there can be multiple cell ranges that are relevant for the answer, thus there can be multiple `Spans` here)
        offsets_in_context:     List of `Span` objects with start and end positions of the answer **in the 
                                context** (i.e. the surrounding text/table of a certain window size).
                                For extractive QA: Character where answer starts => `Answer.offsets_in_document[0].start 
                                For TableQA: Cell where the answer starts (counted from top left to bottom right of table) => `Answer.offsets_in_document[0].start
                                (Note that in TableQA there can be multiple cell ranges that are relevant for the answer, thus there can be multiple `Spans` here) 
        document_id:    ID of the document that the answer was located it (if any))
        meta:   Dict that can be used to associate any kind of custom meta data with the answer. 
                In extractive QA, this will carry the meta data of the document where the answer was found.
    """
    def __post_init__(self):
        # In case offsets are passed as dicts rather than Span objects we convert them here
        # For example, this is used when instantiating an object via from_json()
        if self.offsets_in_document is not None:
            self.offsets_in_document = [Span(**e) if isinstance(e, dict) else e for e in self.offsets_in_document]
        if self.offsets_in_context is not None:
            self.offsets_in_context = [Span(**e) if isinstance(e, dict) else e for e in self.offsets_in_context]
        
        if self.meta is None:
            self.meta = {}
        
        def __lt__(self, other):
            """Enable sorting of Answers by score"""
            return self.score < other.score

        def __str__(self):
            # self.context might be None (therefore not subscriptable)
            if not self.context:
                return f"<Answer: answer='{self.answer}', score={self.score}, context=None>"
            return f"<Answer: answer='{self.answer}', score={self.score}, context='{self.context[:50]}{'...' if len(self.context) > 50 else ''}'>"

        def __repr__(self):
            return f"<Answer {asdict(self)}>"

        def to_dict(self):
            return asdict(self)
        
        @classmethod
        def from_dict(cls, dict: dict):
            return _pydantic_dataclass_from_dict(dict=dict, pydantic_dataclass_type=cls)

        def to_json(self):
            return json.dumps(self, default=pydantic_encoder)

        @classmethod
        def from_json(cls, data):
            if type(data) == str:
                data = json.loads(data)
            return cls.from_dict(data)

def _pydantic_dataclass_from_dict(dict: dict, pydantic_dataclass_type) -> Any:
    """
    Constructs a pydantic dataclass from a dict incl. other nested dataclasses.
    This allows simple de-serialization of pydantic dataclasses from json.
    :param dict: Dict containing all attributes and values for the dataclass.
    :param pydantic_dataclass_type: The class of the dataclass that should be constructed (e.g. Document)
    """
    base_model = pydantic_dataclass_type.__pydantic_model__.parse_obj(dict)
    base_mode_fields = base_model.__fields__

    values = {}
    for base_model_field_name, base_model_field in base_mode_fields.items():
        value = getattr(base_model, base_model_field_name)
        values[base_model_field_name] = value

    dataclass_object = pydantic_dataclass_type(**values)
    return dataclass_object

class NumpyEncoder(json.JSONEncoder): 
    def default(self, obj):
        if isinstance(obj, np.ndarray): 
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
if __name__ == "__main__": 
    d = {
        "content": "My name is Khang", 
        "meta": {
            "Age": 21, 
            "Major": "AI", 
            "Address": "District 7"
        }
    }
    doc = Document.from_dict(d)
    print(doc, type(doc))
        