from typing import Optional

from transformers.models.llama import LlamaTokenizerFast


class DeepseekTokenizerFast(LlamaTokenizerFast):
    def _convert_id_to_token(self, index: int) -> Optional[str]:
        token = self._tokenizer.id_to_token(int(index))
        return token if token is not None else ""
