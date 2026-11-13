"""python -m chesstwin  -> start the web app."""
import argparse
import logging
import threading
import webbrowser


def main():
    ap = argparse.ArgumentParser(prog="chesstwin", description="Play chess against an AI trained on your own Lichess games.")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from other machines")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", help="where profiles are stored (default: ./data or $CHESSTWIN_DATA)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import uvicorn
    from .server import create_app
    app = create_app(data_dir=args.data_dir)
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    print(f"\n  Chess Twin is running at  http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
