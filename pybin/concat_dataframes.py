import polars as pl
import typer

app = typer.Typer()


@app.command()
def concat(p1, p2, out_path):
    pl.concat(
        [pl.read_parquet(p1), pl.read_parquet(p2)], how="vertical"
    ).write_parquet(out_path)


if __name__ == "__main__":
    app()
