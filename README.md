# Fast Math Program Synthesis Tool:

Tool for generating fast approximations of floating point math by (ab)using the floating point representation.

Not yet ready for use.

## Development Environment:

There's a `flake.nix` file for all nixos users.

The python dependencies are managed through `uv`. Setting everything up usually involves 
running a combination of `uv venv; uv sync --all-extrast; source venv/bin/activate`.

To run tests, use `lit tests/filecheck`, there are no pytests yet.

## License:

ffcc - fast float compiler - Copyright (C) 2025 Anton Lydike

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
