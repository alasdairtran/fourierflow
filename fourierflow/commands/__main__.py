from typer import Typer

from fourierflow.commands import download, generate, plot, predict, test, train
from fourierflow.utils import setup_logger

setup_logger()

app = Typer()
app.add_typer(download.app, name='download')
app.add_typer(generate.app, name='generate')
app.add_typer(plot.app, name='plot')
app.add_typer(predict.app, name='predict')
app.add_typer(test.app, name='test')
app.add_typer(train.app, name='train')

if __name__ == '__main__':
    app()
