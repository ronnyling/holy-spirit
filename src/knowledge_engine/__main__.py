def main() -> None:
    from .server import main as server_main

    raise SystemExit(server_main())


if __name__ == "__main__":
    main()
