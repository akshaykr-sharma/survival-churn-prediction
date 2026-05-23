"""PySpark session factory and common helpers."""

from __future__ import annotations

import os
import sys
from typing import Optional

from pyspark.sql import SparkSession

from src.utils.logger import get_logger


def _configure_windows_spark_env() -> None:
    """
    Resolve JAVA_HOME, HADOOP_HOME, and PATH for PySpark on Windows.

    PowerShell $env: assignments are process-local and are NOT inherited by
    subprocesses started in the same session after a SetEnvironmentVariable call.
    This function reads User-scope values directly from the registry so the
    script works regardless of how the shell session was opened.
    """
    # 1. Pull User-scope env vars from the Windows registry
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as _key:
            for _var in ("JAVA_HOME", "HADOOP_HOME"):
                if _var not in os.environ:
                    try:
                        _val, _ = winreg.QueryValueEx(_key, _var)
                        os.environ[_var] = _val
                    except FileNotFoundError:
                        pass
    except Exception:
        pass

    # 2. Auto-detect JAVA_HOME if still missing — search common JDK install roots
    if "JAVA_HOME" not in os.environ:
        for _base in [
            r"C:\Program Files\Eclipse Adoptium",
            r"C:\Program Files\Java",
            r"C:\Program Files\Microsoft",
            r"C:\Program Files\OpenJDK",
        ]:
            try:
                for _entry in sorted(os.listdir(_base), reverse=True):
                    _candidate = os.path.join(_base, _entry)
                    if os.path.isfile(os.path.join(_candidate, "bin", "java.exe")):
                        os.environ["JAVA_HOME"] = _candidate
                        break
            except FileNotFoundError:
                continue
            if "JAVA_HOME" in os.environ:
                break

    # 3. Add JAVA_HOME/bin to PATH so the JVM subprocess can be found
    _java_bin = os.path.join(os.environ.get("JAVA_HOME", ""), "bin")
    if _java_bin and os.path.isdir(_java_bin):
        if _java_bin.lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = _java_bin + os.pathsep + os.environ.get("PATH", "")

    # 4. Add HADOOP_HOME/bin to PATH so hadoop.dll is found by the Windows DLL loader
    _hadoop_home = os.environ.get("HADOOP_HOME", r"C:\hadoop")
    _hadoop_bin = os.path.join(_hadoop_home, "bin")
    if os.path.isfile(os.path.join(_hadoop_bin, "winutils.exe")):
        os.environ["HADOOP_HOME"] = _hadoop_home
        if _hadoop_bin.lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = _hadoop_bin + os.pathsep + os.environ.get("PATH", "")


# On Windows, PySpark defaults to 'python3' which doesn't exist.
# Pin both driver and worker to the interpreter that launched this process.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

if sys.platform == "win32":
    _configure_windows_spark_env()

log = get_logger(__name__)


def get_spark_session(
    app_name: str = "xyz-churn-survival",
    master: str = "local[*]",
    executor_memory: str = "4g",
    driver_memory: str = "4g",
    executor_cores: int = 2,
    shuffle_partitions: int = 50,
    log_level: str = "WARN",
    extra_conf: Optional[dict] = None,
) -> SparkSession:
    """Create or retrieve an active SparkSession with production-sensible defaults."""
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.cores", str(executor_cores))
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.parquet.mergeSchema", "false")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # Vectorized reader rejects INT64 columns written without Parquet logical-type
        # annotations (dates/timestamps stored by some PyArrow versions).
        # Row-based reader applies the schema cast instead of hard-failing.
        .config("spark.sql.parquet.enableVectorizedReader", "false")
    )

    if extra_conf:
        for k, v in extra_conf.items():
            builder = builder.config(k, v)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)

    log.info(
        "SparkSession ready",
        app_name=app_name,
        master=master,
        version=spark.version,
    )
    return spark


def stop_spark(spark: SparkSession) -> None:
    spark.stop()
    log.info("SparkSession stopped")


def read_parquet(spark: SparkSession, path: str):
    """Read parquet with schema inference disabled for speed."""
    log.info("Reading parquet", path=path)
    return spark.read.parquet(path)


def write_parquet(
    df, path: str, mode: str = "overwrite", partitions: Optional[list] = None
) -> None:
    """Write DataFrame as snappy-compressed parquet."""
    log.info("Writing parquet", path=path, mode=mode, partitions=partitions)
    writer = df.write.mode(mode)
    if partitions:
        writer = writer.partitionBy(*partitions)
    writer.parquet(path)
    log.info("Write complete", path=path, rows=df.count())
