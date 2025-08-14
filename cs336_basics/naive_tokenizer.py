from __future__ import annotations
import json
import os
from typing import Iterable, Iterator, Optional, Self
from collections import Counter, defaultdict
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def byte_level_tokenization(word: str) -> tuple[bytes, ...]:
    return tuple([bytes([byte]) for byte in word.encode()])


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]] = None
    ) -> None:
        self.vocab = vocab
        self.vocab_inverted = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_ranked = {merge: i for i, merge in enumerate(merges)}
        if not special_tokens:
            special_tokens = []
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)

        for token in self.special_tokens:
            token_bytes = token.encode()
            if token_bytes not in self.vocab_inverted:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_inverted[token_bytes] = new_id


    @staticmethod
    def _merge_tokens(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        newtoken = b"".join(pair)
        l = len(word)
        i = 0
        output = []
        while i < l - 1:
            if (word[i], word[i + 1]) == pair:
                output.append(newtoken)
                i += 2
            else:
                output.append(word[i])
                i += 1

        if i == l - 1:
            output.append(word[i])

        return tuple(output)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None) -> Self:
        with open(merges_filepath, "r") as f:
            merges = [tuple(token.encode("utf-8") for token in line.rstrip().split(" ")) for line in f]

        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)

        if not special_tokens:
            special_tokens = []

        return cls(vocab, merges, special_tokens)  # type: ignore

    @staticmethod
    def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
        special_tokens = [re.escape(token) for token in special_tokens]
        pattern = f"({'|'.join(special_tokens)})"
        chunks = re.split(pattern, text)

        return chunks

    def _pretokenize(self, text: str) -> Iterator[re.Match[str]]:
        return re.finditer(PAT, text)

    def encode(self, text: str) -> list[int]:
        words_as_tuples = []
        if self.special_tokens:
            splits = self._split_on_special_tokens(text, self.special_tokens)
        else:
            splits = [text]
        for sentence in splits:
            if sentence not in self.special_tokens:
                pretokens: Iterator = self._pretokenize(sentence)
                for word_match in pretokens:
                    word = word_match.group()
                    word_as_byte_list = [bytes([byte]) for byte in word.encode()]
                    word_as_byte_tuple = tuple(word_as_byte_list)
                    words_as_tuples.append(word_as_byte_tuple)
            else:
                words_as_tuples.append((sentence.encode(),))

        token_sequence = []
        for word in words_as_tuples:
            word_tokenized = self.BPE_encode_one_word(word)
            for token in word_tokenized:
                token_sequence.append(self.vocab_inverted[token])

        return token_sequence

    def BPE_encode_one_word(self, word: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if word in self.special_tokens:
            return word
        INF = float("inf")
        min_rank = INF
        min_pair = None
        while True:
            for a, b in zip(word, word[1:]):
                if (a, b) not in self.merges_ranked:
                    continue
                if self.merges_ranked[(a, b)] < min_rank:
                    min_rank = self.merges_ranked.get((a, b), INF)
                    min_pair = (a, b)
            if min_rank < INF and min_pair is not None:
                word = self._merge_tokens(word, min_pair)
                min_rank = INF
                min_pair = None
            else:
                return word

    def byte_encode_one_word(self, word: str) -> tuple[bytes, ...]:
        return tuple([letter.encode() for letter in word])

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            result = self.encode(text)
            for id in result:
                yield id

    def decode(self, ids: list[int]) -> str:
        bytestring = b"".join(self.vocab[id] for id in ids)
        return bytestring.decode(errors="replace")


def make_counts_dicts(text: str, special_tokens=Optional[list[str]]) -> tuple[Counter, Counter, defaultdict]:
    # split on special tokens
    splits = []
    special_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(special_tokens)
    splits = re.split(pattern, text)

    # split by regex and count words
    words_counter: Counter = Counter()
    for segment in splits:
        pretokens = re.finditer(PAT, segment)
        for word_match in pretokens:
            word = word_match.group()
            word_as_bytes_tuple = tuple([bytes([byte]) for byte in word.encode()])
            words_counter[word_as_bytes_tuple] += 1

    pairs_counter: Counter = Counter()
    pairs_positions = defaultdict(set)
    for word, count in words_counter.items():
        for pair in zip(word, word[1:]):
            pairs_counter[pair] += count
            pairs_positions[pair].add(word)

    return words_counter, pairs_counter, pairs_positions


def get_maximal_token_pair(pairs_counts_dict: dict[tuple, int]) -> tuple[bytes, bytes]:
    maximal_token_pair, _ = max(pairs_counts_dict.items(), key=lambda item: (item[1], item[0]))

    return maximal_token_pair


def merge_and_update_all(
    max_pair: tuple[bytes, bytes],
    word: tuple[bytes, ...],
    word_counter: Counter,
    pair_counter: Counter,
    pair_positions: defaultdict,
) -> None:
    count = word_counter[word]

    newtoken = b"".join(max_pair)
    l = len(word)
    i = 0
    output = []
    while i < l - 1:
        if (word[i], word[i + 1]) == max_pair:
            output.append(newtoken)
            if i - 1 >= 0:
                pair_counter[(word[i - 1], newtoken)] += count
                pair_counter[(word[i - 1], word[i])] -= count

            if i < l - 2:  # if there is a token on the right of the merge
                pair_counter[(newtoken, word[i + 2])] += count
                pair_counter[(word[i + 1], word[i + 2])] -= count
            i += 2

        else:
            output.append(word[i])
            i += 1

    if i == l - 1:
        output.append(word[i])

    new_word = tuple(output)

    # update the pair positions, this could be made better, but what the heck
    for newpair in zip(new_word, new_word[1:]):
        pair_positions[newpair].add(new_word)
    for oldpair in zip(word, word[1:]):
        pair_positions[oldpair].discard(word)

    # delete the old one
    del word_counter[word]
    word_counter[new_word] = count


def train_tokenizer(dataset: str | os.PathLike, vocab_size: int, special_tokens: Optional[list[str]]):
    if special_tokens is None:
        special_tokens = []

    if isinstance(dataset, os.PathLike):
        f = open(dataset, "r")
        text = f.read()

    words_counter, pairs_counter, pairs_positions = make_counts_dicts(text, special_tokens)
    f.close()
    vocab = [bytes([i]) for i in range(256)] + [token.encode() for token in set(special_tokens)]
    merges = []

    n_steps = vocab_size - len(vocab)

    for _ in range(n_steps):
        max_pair = get_maximal_token_pair(pairs_counter)
        merges.append(max_pair)
        vocab.append(b"".join(max_pair))

        words_for_merge = list(pairs_positions[max_pair])

        for word in words_for_merge:
            merge_and_update_all(max_pair, word, words_counter, pairs_counter, pairs_positions)

        # now we merged all the max_pair tokens in all words, so we can get rid of it
        del pairs_counter[max_pair]

    vocab_dict = {i: word for i, word in enumerate(vocab)}

    return vocab_dict, merges
