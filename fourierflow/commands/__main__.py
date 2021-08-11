import typer

from fourierflow.commands import download, test, train

app = typer.Typer()
app.add_typer(download.app, name="download")
app.add_typer(train.app, name="train")
app.add_typer(test.app, name="test")

if __name__ == "__main__":
    app()
