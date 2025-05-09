import sys

from synth.main import Main


if __name__ == '__main__':
    main = Main.from_cli(sys.argv)
    main()
