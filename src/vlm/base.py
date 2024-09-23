from abc import ABC, abstractmethod

import os, time, json, duckdb
from torch_snippets import read, resize, Info, in_debug_mode, show, P, PIL, np, Warn, ifnone
from torch_snippets.adapters import np_2_b64

from hashlib import sha256

def to_numpy(image):
    if isinstance(image, (P, str)):
        image = read(image)
    elif isinstance(image, PIL.Image.Image):
        image = np.array(image)
    assert isinstance(image, np.ndarray), f"{type(image)} cannot be sent to downstream functions"
    return image

def hash_image(image):
    image = to_numpy(image)
    image = np.ascontiguousarray(image)
    return sha256(image).hexdigest()

def hash_prompt(prompt:str):
    return sha256(prompt.encode()).hexdigest()

def hash_dict(d):
    dict_str = json.dumps(d, sort_keys=True)
    return sha256(dict_str.encode()).hexdigest()

class VLM(ABC):
    def __init__(self, db=None):
        self.db = os.environ.get('DUCKDB', db)
        self.init_db()

    def init_db(self):
        self.con = duckdb.connect(self.db)
        self.con.execute('''
        CREATE TABLE IF NOT EXISTS Predictions (
            inputs_hash TEXT,
            prompt TEXT,
            kwargs TEXT,
            vlm_name TEXT,
            dataset_name TEXT,
            dataset_row_index INTEGER,
            prediction_value TEXT,
            prediction_duration DECIMAL(6, 3),
            error_string TEXT
        );
        ''')

    @abstractmethod
    def __call__(self, image, prompt, **kwargs):
        pass

    def path_2_b64(self, path, image_size=None):
        if in_debug_mode():
            print(image_type)
            return
        if isinstance(path, (str,P)):
            image = read(path)
            image_type = f'image/{P(path).extn}'
        elif isinstance(path, PIL.Image.Image):
            image = np.array(path)
            image_type = f'image/jpeg'
        else:
            raise NotImplementedError(f"Yet to implement for {type(path)}")
        if image_size:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            image = resize(image, ('at-most', image_size))
        if in_debug_mode():
            Info(f'{image.shape=}')
            show(image)
        return np_2_b64(image), image_type

    def fetch_cache_if_exists(self, image, prompt, **kwargs):
        img_hash = hash_image(image)
        prompt_hash = hash_prompt(prompt)
        dict_hash = hash_dict(kwargs)
        inputs_hash = f'{img_hash}__{prompt_hash}__{dict_hash}'
        with self.con.cursor() as c:
            c.execute(f"SELECT prediction_value FROM Predictions WHERE inputs_hash='{inputs_hash}' and vlm_name='{self.__class__.__name__}'")
            row = c.fetchone()
            if row:
                Info(f'Cache hit for given inputs')
                return row[0], inputs_hash
        return None, inputs_hash

    def make_prediction(self, image, prompt, **kwargs):
        try:
            output = self.predict(image, prompt, **kwargs)
            error = None
        except Exception as e:
            output = None
            error = f'{e}\n{e.__traceback__}'
            Warn(f'Error: {error}')
        return output, error

    def __call__(self, image, prompt, dataset_name=None, dataset_row_id=None, overwrite_cache=False, **kwargs):
        cache, inputs_hash = self.fetch_cache_if_exists(image, prompt, **kwargs)
        if not overwrite_cache and cache:
            Info(f'Returning from cache')
            return cache
        start = time.time()
        output, error = self.make_prediction(image, prompt, **kwargs)
        end = time.time()
        _time = float(f'{end-start:.3f}')
        self.save_prediction(inputs_hash, prompt, kwargs, _time, output, error, dataset_name, dataset_row_id, overwrite_cache)
        return output
    
    def save_prediction(self, inputs_hash, prompt, kwargs, _time, output, error, dataset_name=None, dataset_row_id=None, overwrite_cache=False):
        with self.con.cursor() as c:
            if dataset_name is None or dataset_row_id is None:
                Warn("Calling VLM works best with `dataset_name` and `dataset_row_id`")
            
            kwargs = json.dumps(kwargs, sort_keys=True)

            _insert = (
                "INSERT INTO Predictions (inputs_hash, prompt, kwargs, vlm_name, "
                "dataset_name, dataset_row_index, prediction_value, "
                "prediction_duration, error_string) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            if overwrite_cache:
                Info(f'Overwriting cache for given inputs')
                c.execute(f"DELETE FROM Predictions WHERE inputs_hash='{inputs_hash}'")
            c.execute(
                _insert, 
                (
                    inputs_hash, prompt, kwargs, 
                    self.__class__.__name__, 
                    dataset_name, dataset_row_id, 
                    output, _time, error))
        return output

def set_null(x):
    x = 'NULL' if x is None else f"'{x}'"
    return x
