import re




class TransformerTokenizer:
    def __init__(
            self, 
            encoder_dir:str = "token/encoder_vocab.txt", 
            decoder_dir:str = "token/decoder_vocab.txt"
        ):

        with open(encoder_dir, "r", encoding = "utf-8") as f:
            self.encoder_vocab = f.read()
        with open(decoder_dir, "r", encoding = "utf-8") as f:
            self.decoder_vocab = f.read()

        self.encoder_vocab = self.encoder_vocab.split("\n")
        self.decoder_vocab = self.decoder_vocab.split("\n")

        self.encoder_vocab_to_index = {word: idx for idx, word in enumerate(self.encoder_vocab)}
        self.decoder_vocab_to_index = {word: idx for idx, word in enumerate(self.decoder_vocab)}

        self.encoder_index_to_vocab = {idx: word for idx, word in enumerate(self.encoder_vocab)}
        self.decoder_index_to_vocab = {idx: word for idx, word in enumerate(self.decoder_vocab)}

        # 긴 토큰부터 매칭되도록 정렬
        self.encoder_vocab = sorted(self.encoder_vocab, key=len, reverse=True)
        self.decoder_vocab = sorted(self.decoder_vocab, key=len, reverse=True)

        # 정규 표현식 패턴 생성
        self.encoder_pattern = re.compile("|".join(map(re.escape, self.encoder_vocab)))
        self.decoder_pattern = re.compile("|".join(map(re.escape, self.decoder_vocab)))


    def encode(self, sentence: str, encoder: bool = True):
        """
        문장을 빠르게 토큰 인덱스로 변환
        """
        vocab_to_index = self.encoder_vocab_to_index if encoder else self.decoder_vocab_to_index
        pattern = self.encoder_pattern if encoder else self.decoder_pattern

        tokens = []
        offset = 0

        # 정규 표현식으로 모든 매칭 찾기
        for match in pattern.finditer(sentence):
            start, end = match.span()
            
            # 매칭 안 된 부분을 <UNK>로 채움
            while offset < start:
                tokens.append(vocab_to_index.get("<UNK>", 0))
                offset += 1

            tokens.append(vocab_to_index.get(match.group(), vocab_to_index["<UNK>"]))
            offset = end

        # 남은 부분도 <UNK>로 채움
        while offset < len(sentence):
            tokens.append(vocab_to_index.get("<UNK>", 0))
            offset += 1

        return tokens

    def decode(self, tokens: list, encoder: bool = True):
        index_to_vocab = self.encoder_index_to_vocab if encoder else self.decoder_index_to_vocab
        words = [index_to_vocab.get(token, "<UNK>") for token in tokens]

        return "".join(words).replace("<PAD>", "")

if __name__ == "__main__":
    tt = TransformerTokenizer()
    out = tt.encode("안녕하세요. 저는 abc 입니다.")
    out2 = tt.decode(out)
    print(out)
    print(out2)


