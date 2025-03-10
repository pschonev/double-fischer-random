import dataclasses
import logging
import warnings
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SAWarning
from sqlalchemy.sql import text
from sqlmodel import (
    Session,
    SQLModel,
    create_engine,
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
