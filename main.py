import sys

from emg_pipeline.cli import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "inspect", *sys.argv[1:]]
    main()
