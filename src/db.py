import dataclasses
import logging
import warnings
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SAWarning
from sqlalchemy.sql import text
from sqlmodel import (
    Field,
    PrimaryKeyConstraint,
    Session,
    SQLModel,
    create_engine,
    select,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ParquetDatabase[T: SQLModel]:
    model_class: type[T]
    parquet_file: Path
    engine: Engine = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.engine = create_engine("duckdb:///:memory:")
        SQLModel.metadata.create_all(self.engine)

        # Load data from parquet if available
        if self.parquet_file.exists():
            with Session(self.engine) as session:
                # Load the data from parquet
                session.exec(
                    text(
                        f"INSERT INTO {self.model_class.__tablename__} "  # noqa: S608
                        f"SELECT * FROM read_parquet('{self.parquet_file}')",
                    ),  # type: ignore
                )  # type: ignore
                session.commit()

    def append(self, items: list[T]) -> None:
        """Add new items to the database."""
        # Convert SQLAlchemy warnings to exceptions for this operation
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=SAWarning)

            with Session(self.engine) as session:
                for item in items:
                    try:
                        session.add(item)
                        session.flush()  # This will now raise an exception for warnings
                    except IntegrityError as e:
                        raise ValueError(
                            f"Aborting attempt to add duplicate item: {item}",
                        ) from e
                    except SAWarning as e:
                        raise SAWarning(
                            "Detected duplicate primary key in batch. Aborting append operation. Item:",
                            str(item),
                        ) from e

                session.commit()
                self.save()

    def save(self, codec: str = "zstd") -> None:
        """Save the database to parquet file with specified compression."""
        with Session(self.engine) as session:
            session.exec(
                text(
                    f"COPY {self.model_class.__tablename__} TO '{self.parquet_file}' "
                    f"(FORMAT 'parquet', CODEC '{codec}')",
                ),  # type: ignore
            )  # type: ignore


class Hero(SQLModel, table=True):
    """Example class with composite primary key for testing ParquetDatabase."""

    __tablename__ = "heroes"  # type: ignore

    # Composite primary key
    universe_id: int = Field(primary_key=True)
    hero_code: str = Field(primary_key=True)

    # Regular fields
    name: str = Field(index=True)
    alias_name: str
    age: int = Field(default=0)

    __table_args__ = (PrimaryKeyConstraint("universe_id", "hero_code", name="pk_hero"),)


if __name__ == "__main__":
    db = ParquetDatabase(model_class=Hero, parquet_file=Path("heroes.parquet"))

    # Add some heroes with composite keys
    db.append(
        [
            Hero(
                universe_id=1,
                hero_code="DP",
                name="Deadpond",
                alias_name="Dive Wilson",
            ),
            Hero(
                universe_id=1,
                hero_code="SB",
                name="Spider-Boy",
                alias_name="Pedro Parqueador",
            ),
            Hero(
                universe_id=1,
                hero_code="RM",
                name="Rusty-Man",
                alias_name="Tommy Sharp",
                age=48,
            ),
            Hero(
                universe_id=2,
                hero_code="TR",
                name="Tarantula",
                alias_name="Natalia Roman-on",
            ),
            Hero(
                universe_id=2,
                hero_code="BL",
                name="Black Lion",
                alias_name="Trevor Challa",
            ),
            # Same hero code but different universe - should be allowed
            Hero(
                universe_id=2,
                hero_code="DP",
                name="Dr. Peculiar",
                alias_name="Steve Weird",
                age=36,
            ),
            # Try adding a duplicate - should be skipped
            Hero(
                universe_id=1,
                hero_code="DP",
                name="Deadpond Clone",
                alias_name="Dive Wilson Clone",
            ),
        ],
    )
    db.save()

    # Print the database contents
    with Session(db.engine) as session:
        heroes = session.exec(select(Hero)).all()
        for hero in heroes:
            logging.info(
                f"Hero {hero.universe_id}-{hero.hero_code}: {hero.name} ({hero.alias_name}), Age: {hero.age}",
            )
