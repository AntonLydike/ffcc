{
  description = "xDSL devshell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
        in
          {
            devShells.default = with pkgs; mkShell {
              LD_LIBRARY_PATH = lib.makeLibraryPath [
                stdenv.cc.cc.lib
                zlib
                llvmPackages_20.openmp
                "/run/opengl-driver"
              ];
              NIX_ENFORCE_NO_NATIVE = 0;
              buildInputs = [
                uv
                nodejs_22
                clang_20
                lld_20
		        llvmPackages_20.openmp
		        python312Full
		        llvmPackages_20.libllvm
              ];
            };
          }
    );
}
