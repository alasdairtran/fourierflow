from typer import Typer

from fourierflow.commands import download, generate, test, train
from fourierflow.utils import setup_logger

setup_logger()

app = Typer()
app.add_typer(download.app, name='download')
app.add_typer(train.app, name='train')
app.add_typer(test.app, name='test')
app.add_typer(generate.app, name='generate')

if __name__ == '__main__':
    app()
