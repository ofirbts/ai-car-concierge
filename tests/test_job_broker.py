import time

from backend.job_broker import JobBroker, JobState


def test_job_broker_completes_job():
    broker = JobBroker()

    job_id = broker.submit(lambda _ctx: {"ok": True})
    deadline = time.time() + 2.0
    record = broker.get(job_id)
    while record is not None and record.state in {JobState.PENDING, JobState.RUNNING} and time.time() < deadline:
        time.sleep(0.02)
        record = broker.get(job_id)

    assert record is not None
    assert record.state == JobState.COMPLETED
    assert record.result == {"ok": True}


def test_job_broker_cancels_running_job():
    broker = JobBroker()

    def long_task(ctx):
        for _ in range(50):
            if ctx.is_cancelled():
                return {"cancelled": True}
            time.sleep(0.02)
        return {"done": True}

    job_id = broker.submit(long_task)
    assert broker.cancel(job_id) is True

    deadline = time.time() + 2.0
    record = broker.get(job_id)
    while record is not None and record.state in {JobState.PENDING, JobState.RUNNING} and time.time() < deadline:
        time.sleep(0.02)
        record = broker.get(job_id)

    assert record is not None
    assert record.state == JobState.CANCELLED
