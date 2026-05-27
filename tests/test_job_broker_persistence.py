import time

from backend.database import get_job_record
from backend.job_broker import JobBroker, JobState


def test_job_broker_persists_to_database():
    broker = JobBroker()
    job_id = broker.submit(lambda _ctx: {"value": 1}, metadata={"kind": "unit"})

    deadline = time.time() + 2.0
    record = broker.get(job_id)
    while record is not None and record.state in {JobState.PENDING, JobState.RUNNING} and time.time() < deadline:
        time.sleep(0.02)
        record = broker.get(job_id)

    assert record is not None
    db_record = get_job_record(job_id)
    assert db_record is not None
    assert db_record["state"] == JobState.COMPLETED.value
    assert db_record["result"] == {"value": 1}
