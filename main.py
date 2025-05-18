from typing import Optional

from utils.parser_creation import create_parser
from controllers.data_controller import add_data, summarize
from controllers.train_controller import train_model
from controllers.test_controller import test_model
from utils.logger import get_logger

logger = get_logger(__name__)

def main(argv: Optional[list[str]] = None) -> None:
    """
    Entry point for CLI commands.

    Parameters
    ----------
    argv : list of str, optional
        List of command-line arguments. If None, defaults to sys.argv.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    logger.debug(f"Executing command: {args.command}")
  
    try:
        if args.command == "add_data":
            add_data(args.file_path, args.db, args.start, args.end, args.source_type)
        elif args.command == "train":
            train_model(args.model, bool(args.warm_start), args.db, args.start, args.end)
        elif args.command == "test":
            test_model(args.model, args.db, args.metric, args.source_type)
        elif args.command == "summ":
            summarize(args.db, args.start, args.end)
        else:
            parser.error(f"Unknown command {args.command}")
    except Exception as e:
        logger.exception(f"Command '{args.command}' failed with error:")
        raise

if __name__ == "__main__":
    main()

