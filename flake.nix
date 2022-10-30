{
  description = "A very basic flake";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      # wrap this in another let .. in to add the hydra job only for a single architecture
      output_set = flake-utils.lib.eachDefaultSystem (system:
        let
            pkgs = nixpkgs.legacyPackages.${system};
        in
        rec {
            packages = flake-utils.lib.flattenTree {
                bembel = pkgs.gcc10Stdenv.mkDerivation {
                    name = "Bembel";
                    src = ./.;

                    nativeBuildInputs = [pkgs.cmake];

                    buildInputs = [pkgs.eigen];
                };
            };

            defaultPackage = packages.bembel;
        }
    );
    in
      output_set // { hydraJobs.build."aarch64-linux" = output_set.defaultPackage."aarch64-linux"; };
}
