from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from time import time
import click
import os
from typing import BinaryIO, Iterator, Optional
import regex as re

from tqdm import tqdm


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def make_counts_dicts(text, special_tokens=Optional[list[str]]):
    # split on special tokens
    splits = []
    special_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(special_tokens)
    splits = re.split(pattern, text)

    # split by regex and count words
    words_counter = Counter()
    for segment in splits:
        pretokens = re.finditer(PAT, segment)
        for word_match in pretokens:
            word = word_match.group()
            word_as_bytes_tuple = tuple([bytes([byte]) for byte in word.encode()])
            words_counter[word_as_bytes_tuple] += 1

    pairs_counter = Counter()
    pairs_positions = defaultdict(set)
    for word, count in words_counter.items():
        for pair in zip(word, word[1:]):
            pairs_counter[pair] += count
            pairs_positions[pair].add(word)

    return words_counter, pairs_counter, pairs_positions


def process_chunk(args):
    # process one chunk in parallel
    f, start, end = args
    with open(f, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    return make_counts_dicts(chunk, ["<|endoftext|>"])


def get_maximal_token_pair(pairs_counts_dict: dict[tuple, int]) -> tuple[bytes, bytes]:
    maximal_token_pair, _ = max(pairs_counts_dict.items(), key=lambda item: (item[1], item[0]))

    return maximal_token_pair


def merge_and_update_all(max_pair, word, word_counter, pair_counter, pair_positions):
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


@click.command()
@click.argument(
    "dataset",
    type=click.Path(exists=True),
    default="/Users/tomas.vavra/python-projects/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
)
@click.option("--vocab_size", type=click.INT, default=1000)
@click.option("--special_tokens", type=click.STRING, default="['<|endoftext|>']")
@click.option("--n_chunks", type=click.INT, default=40)
@click.option("--n_proc", type=click.INT, default=8)
def train_tokenizer(**kwargs):
    if isinstance(kwargs["special_tokens"], str):
        special_tokens = eval(kwargs["special_tokens"])
    else:
        special_tokens = kwargs["special_tokens"]

    with open(kwargs["dataset"], "rb") as f:
        boundaries = find_chunk_boundaries(f, kwargs["n_chunks"], "<|endoftext|>".encode("utf-8"))

    ###################################
    ##### MULTIPROCESS PART ###########
    ###################################
    jobs = ((kwargs["dataset"], s, e) for (s, e) in zip(boundaries[:-1], boundaries[1:]))

    total_chunks = len(boundaries) - 1

    word_counter = Counter()
    pair_counter = Counter()
    pair_positions = defaultdict(set)

    start_t = time()
    with Pool(processes=kwargs["n_proc"]) as pool:
        for chunk_word_counter, chunk_pair_counter, chunk_pair_positions in tqdm(
            pool.imap_unordered(process_chunk, jobs, chunksize=40), total=total_chunks
        ):
            word_counter.update(chunk_word_counter)
            pair_counter.update(chunk_pair_counter)
            for k, s in chunk_pair_positions.items():
                pair_positions[k].update(s)
    print(f"Processed the file and built the counts in {time()-start_t}s")

    ###################################

    vocab = [bytes([i]) for i in range(256)] + [token.encode() for token in set(special_tokens)]
    merges = []

    n_steps = kwargs["vocab_size"] - len(vocab)

    for _ in tqdm(range(n_steps)):
        max_pair = get_maximal_token_pair(pair_counter)
        merges.append(max_pair)
        vocab.append(b"".join(max_pair))

        words_for_merge = list(pair_positions[max_pair])

        for word in words_for_merge:
            merge_and_update_all(max_pair, word, word_counter, pair_counter, pair_positions)

        # now we merged all the max_pair tokens in all words, so we can get rid of it
        del pair_counter[max_pair]

    vocab_dict = {i: word for i, word in enumerate(vocab)}
    print(f"The whole thing took {time() - start_t}")

    return vocab_dict, merges


if __name__ == "__main__":
    train_tokenizer()
