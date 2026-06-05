import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.treinar_regressao_linear import main


if __name__ == "__main__":
    main()
