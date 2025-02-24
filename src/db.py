import dataclasses
from pathlib import Path
from typing import Generic, Type, TypeVar

from sqlalchemy import Sequence
from sqlalchemy.engine import Engine
from sqlalchemy.sql import text
from sqlmodel import Session, SQLModel, create_engine, Field

T = TypeVar("T", bound=SQLModel)


@dataclasses.dataclass
class ParquetDatabase(Generic[T]):
    model_class: Type[T]
    parquet_file: Path
    engine: Engine = dataclasses.field(init=False)

    def __post_init__(self):
        self.engine = create_engine("duckdb:///:memory:")
        SQLModel.metadata.create_all(self.engine)

        # Load data from parquet if available and table is empty
        if self.parquet_file.exists():
            with Session(self.engine) as session:
                result = session.exec(
                    text(f"SELECT COUNT(*) FROM {self.model_class.__tablename__}")
                ).first()
                if result == 0:
                    session.exec(
                        text(
                            f"INSERT INTO {self.model_class.__tablename__} "
                            f"SELECT * FROM read_parquet('{self.parquet_file}')"
                        )
                    )
                    session.commit()

    def append(self, *items: T) -> None:
        with Session(self.engine) as session:
            for item in items:
                session.add(item)
            session.commit()

            # Save to parquet after successful commit
            session.exec(
                text(
                    f"COPY {self.model_class.__tablename__} TO '{self.parquet_file}' "
                    "(FORMAT 'parquet', CODEC 'zstd')"
                )
            )


def id_field(table_name: str):
    sequence = Sequence(f"{table_name}_id_seq")
    return Field(
        default=None,
        primary_key=True,
        sa_column_args=[sequence],
        sa_column_kwargs={"server_default": sequence.next_value()},
    )


class Hero(SQLModel, table=True):
    id: int | None = id_field("hero")
    name: str
    secret_name: str
    age: int | None = None


if __name__ == "__main__":
    db = ParquetDatabase[Hero](model_class=Hero, parquet_file=Path("data.parquet"))

    # Add some heroes
    db.append(
        Hero(name="Deadpond", secret_name="Dive Wilson"),
        Hero(name="Spider-Boy", secret_name="Pedro Parqueador"),
        Hero(name="Rusty-Man", secret_name="Tommy Sharp", age=48),
    )

    # print the db
    with Session(db.engine) as session:
        heroes = session.query(Hero).all()
        for hero in heroes:
            print(hero)
