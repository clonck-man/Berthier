from mistral_common.protocol.instruct.messages import (
    UserMessage,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest


class CustomTokenizer:
    def __init__(self, model_name="test", is_tekken=True, is_mm=True):
        self.tokenizer = MistralTokenizer.v3(is_tekken=is_tekken, is_mm=is_mm)
        self.model_name = model_name

    def get_tokens_from_string(self, input_string):
        tokenized = self.tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[UserMessage(content=input_string)],
                model=self.model_name
            )
        )
        return tokenized.tokens


def main():
    # Création de l'objet tokenizer
    custom_tokenizer = CustomTokenizer()

    # Exemple d'utilisation pour obtenir les tokens à partir d'une chaîne de caractères
    input_string = "What's the weather like today in Paris?"
    tokens = custom_tokenizer.get_tokens_from_string(input_string)

    print("Tokens:", tokens)

 
if __name__ == "__main__":
    main()
