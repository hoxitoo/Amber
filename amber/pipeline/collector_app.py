from amber.common.logging import setup_logging


def main() -> None:
    setup_logging()
    print("collector_app started")


if __name__ == "__main__":
    main()
