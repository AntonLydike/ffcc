import sys

from ffcc.main import Main


if __name__ == "__main__":
    main = Main.from_cli(sys.argv)
    main()
