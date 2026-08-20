from gradio import Server
from gradio.data_classes import FileData

from argparse import ArgumentParser



async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == '__main__':
    parser = ArgumentParser(description='muscriptor.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    args = parser.parse_args()
    
    app.launch(share=args.share_enabled, show_error=True)
