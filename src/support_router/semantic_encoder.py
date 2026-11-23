"""Frozen, local MiniLM sentence embeddings; no customer text is sent externally.

Encoder weights: sentence-transformers/all-MiniLM-L6-v2, Apache-2.0.
The encoder is NOT fine-tuned on Bitext or the diagnostic sentences.
"""
import hashlib
import json
import platform
import urllib.request
from pathlib import Path
from threading import Lock

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .paths import encoder_dir

REPO = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
PINNED_FILES = {"model.onnx": "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21",
                "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"}
COMPUTATION = 'mean_pool_qint8_single_text_v2_arch_checked'
_ENGINES = {}
_ENGINE_LOCK = Lock()
_VECTORS = {}

def ensure_encoder(directory=None, *, local_only=False):
    directory = Path(directory) if directory else encoder_dir()
    manifest_path = directory/'encoder_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Encoder manifest missing: {manifest_path}. Run the notebook setup cell.')
    manifest = json.loads(manifest_path.read_text())
    machine = platform.machine().lower()
    if machine not in {'x86_64','amd64'}:
        raise RuntimeError('Pinned MiniLM quantized AVX2 model is x86_64-only in this project; run semantic candidates on Colab x86_64')
    if manifest['model'] != REPO or manifest['revision'] != REVISION:
        raise ValueError('Encoder identity mismatch')
    for filename, expected in PINNED_FILES.items():
        if manifest["files"][filename]["sha256"] != expected:
            raise ValueError(f"Unexpected encoder identity: {filename}")
        path = directory/filename
        if not path.exists():
            if local_only:
                raise FileNotFoundError(f'Local encoder file missing: {filename}')
            info = manifest['files'][filename]
            url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{info["source"]}'
            temporary = path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(url,timeout=120) as response, temporary.open('wb') as output:
                while chunk := response.read(1024*1024):
                    output.write(chunk)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest['files'][filename]['sha256']:
            raise ValueError(f'Encoder file hash mismatch: {filename}')
    return directory

def _engine(directory, *, local_only=False):
    key = str(Path(directory).resolve())
    if key in _ENGINES:
        return key, _ENGINES[key]
    with _ENGINE_LOCK:
        if key not in _ENGINES:
            directory = ensure_encoder(directory, local_only=local_only)
            import onnxruntime as ort
            from tokenizers import Tokenizer
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(directory/'model.onnx'), sess_options=options,
                                          providers=['CPUExecutionProvider'])
            tokenizer = Tokenizer.from_file(str(directory/'tokenizer.json'))
            tokenizer.enable_truncation(max_length=256)
            tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
            cache = directory/'embeddings_cache.npz'
            if not local_only and cache.exists():
                stored = np.load(cache,allow_pickle=False)
                if str(stored['revision']) == REVISION and 'computation' in stored.files and str(stored['computation']) == COMPUTATION:
                    _VECTORS[key] = {str(k):v for k,v in zip(stored['keys'],stored['vectors'])}
            _VECTORS.setdefault(key,{})
            _ENGINES[key] = session,tokenizer
    return key, _ENGINES[key]

def _key(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def encode(texts, directory=None, batch_size=64):
    directory = Path(directory) if directory else encoder_dir()
    key,(session,tokenizer) = _engine(directory)
    texts = list(texts)
    if not texts:
        return np.empty((0,384),dtype=np.float32)
    cache = _VECTORS[key]
    missing = list(dict.fromkeys(t for t in texts if _key(t) not in cache))
    # Dynamic quantization changes scales with batch composition. Encode each
    # sentence alone so notebook batches and interactive requests are identical.
    for text in missing:
        cache[_key(text)] = _vector(tokenizer.encode(text), session)
    return np.stack([cache[_key(t)] for t in texts])

def _vector(tokens, session):
    inputs = {name: np.asarray([values], dtype=np.int64) for name, values in
              (("input_ids", tokens.ids), ("attention_mask", tokens.attention_mask),
               ("token_type_ids", tokens.type_ids))}
    hidden = session.run(["last_hidden_state"], inputs)[0][0]
    vector = np.average(hidden, axis=0, weights=inputs["attention_mask"][0].astype(np.float32))
    return (vector / max(np.linalg.norm(vector), 1e-9)).astype(np.float32)


def serving_vector(text, directory):
    """Local inference only. Do not retain vectors from incoming tickets."""
    _, (session, tokenizer) = _engine(directory, local_only=True)
    tokens = tokenizer.encode(text)
    if tokens.overflowing:
        return None
    return _vector(tokens, session)


def save_cache(directory=None):
    directory = Path(directory) if directory else encoder_dir()
    key = str(directory.resolve())
    cache = _VECTORS.get(key,{})
    if cache:
        np.savez_compressed(directory/'embeddings_cache.npz',revision=np.array(REVISION),computation=np.array(COMPUTATION),
                            keys=np.array(list(cache)),vectors=np.stack(list(cache.values())))

def was_truncated(text, directory=None):
    directory=Path(directory) if directory else encoder_dir()
    _,(_,tokenizer)=_engine(directory)
    return bool(tokenizer.encode(text).overflowing)

class FrozenSentenceEncoder(TransformerMixin,BaseEstimator):
    def __init__(self,batch_size=64):
        self.batch_size = batch_size

    def fit(self,X,y=None):
        self.n_features_out_ = 384
        return self

    def transform(self,X):
        return encode(X,batch_size=self.batch_size)

    def get_feature_names_out(self,input_features=None):
        return np.array([f'latent_dimension_{i}' for i in range(384)])
