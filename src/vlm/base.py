from abc import ABC, abstractmethod
import os, duckdb, json, traceback, time
from hashlib import sha256
from torch_snippets import (
    read,
    resize,
    Info,
    in_debug_mode,
    show,
    P,
    np,
    PIL,
    Warn,
    ifnone,
)
from torch_snippets.adapters import np_2_b64


def to_numpy(image):
    if isinstance(image, (P, str)):
        image = read(image)
    elif isinstance(image, PIL.Image.Image):
        image = np.array(image)
    assert isinstance(
        image, np.ndarray
    ), f"{type(image)} cannot be sent to downstream functions"
    return image


def hash_image(image):
    image = to_numpy(image)
    image = np.ascontiguousarray(image)
    return sha256(image).hexdigest()


def hash_prompt(prompt: str):
    return sha256(prompt.encode()).hexdigest()


def hash_dict(d):
    dict_str = json.dumps(d, sort_keys=True)
    return sha256(dict_str.encode()).hexdigest()


class VLM(ABC):
    def __init__(self, *, db=None, name=None):
        self.db = os.environ.get("DUCKDB", db)
        self.init_db()
        if name is None and "Custom" in self.__class__.__name__:
            raise ValueError("Custom Models need a name as argument for DB caching")
        self.name = ifnone(name, self.__class__.__name__)

    def init_db(self):
        self.con = duckdb.connect(self.db)
        self.con.execute(
            """
        CREATE TABLE IF NOT EXISTS Predictions (
            inputs_hash TEXT,
            prompt TEXT,
            kwargs TEXT,
            vlm_name TEXT,
            dataset_name TEXT,
            item_name TEXT,
            prediction_value TEXT,
            prediction_duration DECIMAL(6, 3),
            error_string TEXT
        );
        """
        )

    @abstractmethod
    def predict(self, image, prompt, **kwargs):
        pass

    def path_2_b64(self, path, image_size=None):
        if in_debug_mode():
            print(image_type)
            return
        if isinstance(path, (str, P)):
            image = read(path)
            image_type = f"image/{P(path).extn}"
        elif isinstance(path, PIL.Image.Image):
            image = np.array(path)
            image_type = f"image/jpeg"
        else:
            raise NotImplementedError(f"Yet to implement for {type(path)}")
        if image_size:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            image = resize(image, ("at-most", image_size))
        if in_debug_mode():
            Info(f"{image.shape=}")
            show(image)
        return np_2_b64(image), image_type

    def fetch_cache_if_exists(self, image, prompt, **kwargs):
        img_hash = hash_image(image)
        prompt_hash = hash_prompt(prompt)
        dict_hash = hash_dict(kwargs)
        inputs_hash = f"{img_hash}__{prompt_hash}__{dict_hash}"
        with self.con.cursor() as c:
            _vlm_name = getattr(self, "name", self.__class__.__name__)
            c.execute(
                f"SELECT prediction_value, error_string FROM Predictions WHERE inputs_hash='{inputs_hash}' and vlm_name='{_vlm_name}'"
            )
            row = c.fetchone()
            if row:
                Info(f"Cache hit for given inputs")
                output, error = row
                return (output, error), inputs_hash
        return (None, None), inputs_hash

    def make_prediction(self, image, prompt, **kwargs):
        try:
            output = self.predict(image, prompt, **kwargs)
            error = None
        except Exception as e:
            output = None
            _tb = traceback.format_exc()
            error = f"{e}\n{_tb}"
            Warn(f"Error: {error}")
        return output, error

    def __call__(
        self,
        image,
        prompt,
        dataset_name: str,
        item_name: str,
        overwrite_cache=False,
        **kwargs,
    ):
        (output, error), inputs_hash = self.fetch_cache_if_exists(
            image, prompt, **kwargs
        )
        if output and (not overwrite_cache):
            Info(f"Returning from cache")
            return output
        start = time.time()
        output, error = self.make_prediction(image, prompt, **kwargs)
        end = time.time()
        _time = float(f"{end-start:.3f}")
        self.save_prediction(
            inputs_hash,
            prompt,
            kwargs,
            _time,
            output,
            error,
            dataset_name,
            item_name,
            overwrite_cache,
        )
        return output

    def save_prediction(
        self,
        inputs_hash,
        prompt,
        kwargs,
        _time,
        output,
        error,
        dataset_name=None,
        item_name=None,
        overwrite_cache=False,
    ):
        with self.con.cursor() as c:
            if dataset_name is None or item_name is None:
                Warn("Calling VLM works best with `dataset_name` and `item_name`")

            kwargs = json.dumps(kwargs, sort_keys=True)

            _insert = (
                "INSERT INTO Predictions (inputs_hash, prompt, kwargs, vlm_name, "
                "dataset_name, item_name, prediction_value, "
                "prediction_duration, error_string) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            _vlm_name = getattr(self, "name", self.__class__.__name__)
            if overwrite_cache:
                Info(f"Overwriting cache for given inputs")
                c.execute(
                    f"DELETE FROM Predictions WHERE inputs_hash='{inputs_hash}' and vlm_name='{_vlm_name}'"
                )
                Info(f"Deleted {c.rowcount}")
            c.execute(
                _insert,
                (
                    inputs_hash,
                    prompt,
                    kwargs,
                    _vlm_name,
                    dataset_name,
                    item_name,
                    output,
                    _time,
                    error,
                ),
            )
        return output


def set_null(x):
    x = "NULL" if x is None else f"'{x}'"
    return x
