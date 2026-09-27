import time
import re
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models, similarities

class NLPService:
    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [token.strip() for token in clean_text.split() if len(token.strip()) > 1]
        return tokens
