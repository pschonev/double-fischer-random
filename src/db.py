import dataclasses
from pathlib import Path
from typing import Type

from sqlalchemy import Sequence, UniqueConstraint
from sqlalchemy.sql import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, create_engine, select


@dataclasses.dataclass
class ParquetDatabase[T: SQLModel]:
    model_class: Type[T]
    parquet_file: Path
    engine: Engine = dataclasses.field(init=False)

    def __post_init__(self):
        self.engine = create_engine("duckdb:///:memory:")
        SQLModel.metadata.create_all(self.engine)

        # Load data from parquet if available
        if self.parquet_file.exists():
            with Session(self.engine) as session:
                # Load the data from parquet
                session.exec(
                    text(
                        f"INSERT INTO {self.model_class.__tablename__} "
                        f"SELECT * FROM read_parquet('{self.parquet_file}')"
                    )
                )
                session.commit()

    def append(self, *items: T) -> None:
        """Add new items to the database."""
        with Session(self.engine) as session:
            # Find the maximum ID to ensure new items get proper IDs
            result = session.exec(
                text(f"SELECT MAX(id) FROM {self.model_class.__tablename__}")
            ).first()
            next_id = (result[0] if result and result[0] is not None else 0) + 1

            for item in items:
                # Set ID if not already set
                if getattr(item, "id", None) is None:
                    setattr(item, "id", next_id)

                try:
                    session.add(item)
                    session.flush()  # This will check constraints without committing
                    next_id += 1  # Only increment ID after successful addition
                except IntegrityError:
                    session.rollback()  # Roll back the failed operation
                    print(f"Skipped duplicate item: {item}")
                    continue

            session.commit()
            self.save()

    def save(self, codec: str = "zstd") -> None:
        """Save the database to parquet file with specified compression."""
        with Session(self.engine) as session:
            session.exec(
                text(
                    f"COPY {self.model_class.__tablename__} TO '{self.parquet_file}' "
                    f"(FORMAT 'parquet', CODEC '{codec}')"
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

    name: str = Field(index=True)
    secret_name: str
    age: int = Field(default=0)

    # Add a unique constraint on name and age
    __table_args__ = (UniqueConstraint("name", "age", name="unique_name_age"),)


if __name__ == "__main__":
    db = ParquetDatabase(model_class=Hero, parquet_file=Path("data.parquet"))

    # Add some heroes
    db.append(
        Hero(name="Deadpond", secret_name="Dive Wilson"),
        Hero(name="Spider-Boy", secret_name="Pedro Parqueador"),
        Hero(name="Rusty-Man", secret_name="Tommy Sharp", age=48),
        Hero(name="Tarantula", secret_name="Natalia Roman-on"),
        Hero(name="Black Lion", secret_name="Trevor Challa"),
        Hero(name="Dr. Weird", secret_name="Steve Weird", age=36),
        Hero(name="Dr. Weird", secret_name="Steve Weird", age=12),
    )
    db.save()

    # print the db
    with Session(db.engine) as session:
        heroes = session.exec(select(Hero)).all()
        for hero in heroes:
            print(hero)
