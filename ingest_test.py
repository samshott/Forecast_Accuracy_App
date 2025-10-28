from datetime import datetime, timezone, timedelta
from pathlib import Path
from src.config import load_settings
from src.ingest_forecast_archive import ingest_ndfd_archive

settings = load_settings()
now = datetime.now(timezone.utc)
for issue in [now - timedelta(hours=6), now - timedelta(hours=12), now - timedelta(hours=18)]:
    print('ingesting issue', issue)
    results = ingest_ndfd_archive(issue, variables=['TMAX'], settings=settings, bbox=(24,-125,50,-66), write_parquet=True)
    print(results)
    path = settings.data.forecast_dir / 'TMAX'
    print('files', list(Path(path).glob('*.parquet')))
