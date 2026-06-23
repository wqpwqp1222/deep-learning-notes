import dnnlpy.tokenizers as tk
from dnnlpy.tokenizers import BPETokenizer, Tokenizer


def test_tokenizer_exports_bpe_tokenizer():
    assert hasattr(tk, 'BPETokenizer')


def test_bpe_tokenizer_implements_base_api():
    tokenizer = BPETokenizer.from_text(
        'low lower lowest low lower lowest',
        vocab_size=32,
        min_frequency=2,
    )

    ids = tokenizer.encode('low lowest')

    assert isinstance(tokenizer, Tokenizer)
    assert '<bos>' not in tokenizer.vocab
    assert tokenizer.decode(ids) == 'low lowest'
    assert isinstance(tokenizer.encode('low'), list)


def test_bpe_tokenizer_batch_api():
    tokenizer = BPETokenizer.from_text(
        'low lower lowest low lower lowest',
        vocab_size=32,
        min_frequency=2,
    )

    batch_ids = tokenizer.encode_batch(['low', 'lowest'])

    assert all(isinstance(ids, list) for ids in batch_ids)
    assert tokenizer.decode_batch(batch_ids) == ['low', 'lowest']


def test_bpe_tokenizer_from_text_accepts_list():
    tokenizer = BPETokenizer.from_text(
        ['low lower lowest', 'low lower lowest'],
        vocab_size=32,
        min_frequency=2,
    )

    assert tokenizer.decode(tokenizer.encode('low')) == 'low'


def test_bpe_tokenizer_from_text_uses_custom_unknown_token():
    tokenizer = BPETokenizer.from_text(
        'low lower lowest low lower lowest',
        vocab_size=32,
        min_frequency=2,
        unk_token='<missing>',
    )

    ids = tokenizer.encode('low xyz')

    assert tokenizer.unk_token == '<missing>'
    assert tokenizer.special_tokens == ['<missing>']
    assert tokenizer.unk_id in ids
    assert tokenizer.decode(ids) == 'low'
    assert (
        tokenizer.decode(ids, skip_special_tokens=False)
        == 'low <missing><missing><missing>'
    )
