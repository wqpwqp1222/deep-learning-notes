import pytest

import dnnlpy
import dnnlpy.tokenizers as tk
from dnnlpy.tokenizers import CharacterTokenizer, Tokenizer, WordTokenizer


def test_tokenizer_exports_traditional_tokenizers():
    assert dnnlpy.tokenizers is tk

    for name in [
        'CharacterTokenizer',
        'Tokenizer',
        'WordTokenizer',
    ]:
        assert hasattr(tk, name)


def test_tokenizer_base_lookup_helpers():
    tokenizer = CharacterTokenizer({'<unk>': 0, 'a': 1, 'b': 2})

    assert tokenizer.unk_id == 0
    assert tokenizer.token2id('missing') == tokenizer.unk_id
    assert tokenizer.lookup_indices(['a', 'missing', 'b']) == [1, 0, 2]
    assert tokenizer.lookup_tokens([1, 0, 2]) == ['a', '<unk>', 'b']

    with pytest.raises(KeyError, match='Unknown token ID'):
        tokenizer.id2token(99)


def test_tokenizer_base_rejects_missing_unknown_token():
    with pytest.raises(ValueError, match='is not in vocab'):
        CharacterTokenizer({'a': 0})


def test_tokenizer_add_special_tokens():
    tokenizer = CharacterTokenizer({'<unk>': 0, 'a': 1})

    added_count = tokenizer.add_special_tokens(['<bos>', '<eos>', '<bos>'])

    assert added_count == 2
    assert tokenizer.special_tokens == ['<unk>', '<bos>', '<eos>']
    assert tokenizer.special_token_ids == {
        tokenizer.token2id('<unk>'),
        tokenizer.token2id('<bos>'),
        tokenizer.token2id('<eos>'),
    }


def test_tokenizer_repr_shows_vocab_and_special_token_metadata():
    tokenizer = CharacterTokenizer({'<unk>': 0, 'a': 1})

    assert repr(tokenizer) == (
        "CharacterTokenizer(vocab_size=2, unk_token='<unk>', special_tokens=['<unk>'])"
    )


def test_character_tokenizer_implements_base_api():
    tokenizer = CharacterTokenizer.from_text('banana')

    assert isinstance(tokenizer, Tokenizer)
    assert '<bos>' not in tokenizer.vocab
    assert tokenizer.decode(tokenizer.encode('banana')) == 'banana'
    assert tokenizer.encode('banana') == [
        tokenizer.token2id('b'),
        tokenizer.token2id('a'),
        tokenizer.token2id('n'),
        tokenizer.token2id('a'),
        tokenizer.token2id('n'),
        tokenizer.token2id('a'),
    ]


def test_character_tokenizer_batch_api():
    tokenizer = CharacterTokenizer.from_text('banana')

    batch_ids = tokenizer.encode_batch(['ban', 'ana'])

    assert all(isinstance(ids, list) for ids in batch_ids)
    assert tokenizer.decode_batch(batch_ids) == ['ban', 'ana']


def test_character_tokenizer_from_text_accepts_list():
    tokenizer = CharacterTokenizer.from_text(['ban', 'ana'])

    assert tokenizer.decode(tokenizer.encode('banana')) == 'banana'


def test_character_tokenizer_from_text_uses_custom_unknown_token():
    tokenizer = CharacterTokenizer.from_text('banana', unk_token='?')

    assert tokenizer.unk_token == '?'
    assert tokenizer.special_tokens == ['?']
    assert tokenizer.encode('band')[-1] == tokenizer.unk_id
    assert tokenizer.decode(tokenizer.encode('band')) == 'ban'
    assert (
        tokenizer.decode(tokenizer.encode('band'), skip_special_tokens=False) == 'ban?'
    )


def test_character_tokenizer_decodes_special_tokens():
    tokenizer = CharacterTokenizer.from_text('banana')
    tokenizer.add_special_tokens(['<bos>', '<eos>'])

    ids = [
        tokenizer.token2id('<bos>'),
        tokenizer.token2id('b'),
        tokenizer.token2id('<unk>'),
        tokenizer.token2id('a'),
        tokenizer.token2id('<eos>'),
    ]

    assert tokenizer.decode(ids) == 'ba'
    assert tokenizer.decode(ids, skip_special_tokens=False) == '<bos>b<unk>a<eos>'


def test_character_tokenizer_init_uses_vocab():
    tokenizer = CharacterTokenizer({'<unk>': 0, 'a': 1, 'b': 2})

    assert tokenizer.encode('abc') == [1, 2, 0]


def test_word_tokenizer_implements_base_api():
    tokenizer = WordTokenizer.from_text('deep learning notes')

    ids = tokenizer.encode('deep unknown notes')

    assert isinstance(tokenizer, Tokenizer)
    assert '<bos>' not in tokenizer.vocab
    assert ids[1] == tokenizer.token2id('<unk>')
    assert tokenizer.decode(ids) == 'deep notes'
    assert tokenizer.decode(ids, skip_special_tokens=False) == 'deep <unk> notes'
    assert tokenizer.encode('deep') == [tokenizer.token2id('deep')]


def test_word_tokenizer_batch_api():
    tokenizer = WordTokenizer.from_text('deep learning notes')

    batch_ids = tokenizer.encode_batch(['deep learning', 'notes unknown'])

    assert all(isinstance(ids, list) for ids in batch_ids)
    assert tokenizer.decode_batch(batch_ids) == ['deep learning', 'notes']
    assert tokenizer.decode_batch(batch_ids, skip_special_tokens=False) == [
        'deep learning',
        'notes <unk>',
    ]


def test_word_tokenizer_from_text_accepts_list():
    tokenizer = WordTokenizer.from_text(['deep learning', 'notes'])

    assert tokenizer.decode(tokenizer.encode('deep notes')) == 'deep notes'


def test_word_tokenizer_from_text_uses_custom_unknown_token():
    tokenizer = WordTokenizer.from_text('deep learning notes', unk_token='<missing>')

    ids = tokenizer.encode('deep unknown notes')

    assert tokenizer.unk_token == '<missing>'
    assert tokenizer.special_tokens == ['<missing>']
    assert ids[1] == tokenizer.unk_id
    assert tokenizer.decode(ids) == 'deep notes'
    assert tokenizer.decode(ids, skip_special_tokens=False) == 'deep <missing> notes'


def test_word_tokenizer_decodes_special_tokens():
    tokenizer = WordTokenizer.from_text('deep learning notes')
    tokenizer.add_special_tokens(['<bos>', '<eos>'])

    ids = [
        tokenizer.token2id('<bos>'),
        tokenizer.token2id('deep'),
        tokenizer.token2id('<unk>'),
        tokenizer.token2id('notes'),
        tokenizer.token2id('<eos>'),
    ]

    assert tokenizer.decode(ids) == 'deep notes'
    assert (
        tokenizer.decode(ids, skip_special_tokens=False)
        == '<bos> deep <unk> notes <eos>'
    )


def test_word_tokenizer_init_uses_vocab():
    tokenizer = WordTokenizer({'<unk>': 0, 'deep': 1, 'notes': 2})

    assert tokenizer.encode('deep learning notes') == [1, 0, 2]
