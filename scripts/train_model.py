from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader
from amber.pipeline.train_app import MIN_TRAIN_ROWS, NotEnoughData, run_training


if __name__ == "__main__":
    config = ConfigLoader(Path.cwd()).load_yaml("config/amber.yaml")
    datasets_root = Path(config["storage"]["datasets_dir"])
    models_root = Path(config["storage"]["models_dir"])
    logs_root = Path(config["storage"]["logs_dir"])

    try:
        result = run_training(config, datasets_root, models_root, logs_root)
    except NotEnoughData as exc:
        print(
            "Недостаточно данных для обучения.\n"
            f"В датасете {exc.rows} строк (нужно >= {MIN_TRAIN_ROWS}).\n\n"
            "Что делать:\n"
            "  1) Нажми «REST-backfill истории» — подтянет ~1000 свечей на символ.\n"
            "  2) Затем «Датасет» → «Обучение» (или снова «Полный цикл»).\n"
            "  3) Либо дай WS-коллектору поработать 1-2 часа и повтори.\n\n"
            "Это не ошибка — модели пока не на чем учиться.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    print(result)
