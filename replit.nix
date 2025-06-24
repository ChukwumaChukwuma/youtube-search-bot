
{ pkgs }: {
  deps = [
    pkgs.openssl
    pkgs.pkg-config
    pkgs.which
    # Remove playwright-driver for lighter deployment
    # pkgs.playwright-driver
  ];
}
