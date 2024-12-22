from pathlib import Path

dir_path = Path(__file__).parent.as_posix()

DEFAULT_DATA_DIR = "/cache/data/"
DEFAULT_MODEL_DIR = "/cache/pretrained/"
DEFAULT_TOKENIZER_DIR = Path(dir_path).parent.joinpath("assets/tokenizer/")
DEFAULT_CSV_SUBDIR = "csv"
DEFAULT_TAR_SUBDIR = "tar"

DEFAULT_LABEL_FILENAME = "classnames.txt"
