from app import cache


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.ttls[key] = ttl

    def delete(self, key):
        self.values.pop(key, None)
        self.ttls.pop(key, None)

    def ping(self):
        return True

    def incr(self, key):
        current = int(self.values.get(key, "0")) + 1
        self.values[key] = str(current)
        return current

    def expire(self, key, ttl):
        self.ttls[key] = ttl


def test_knowledge_status_cache_round_trip_and_invalidation(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(cache, "redis_client", fake_redis)
    monkeypatch.setattr(cache, "REDIS_CACHE_ENABLED", True)

    payload = {"is_current": True, "document_count": 2}
    assert cache.set_cached_json("test-key", payload, 30) is True
    assert cache.get_cached_json("test-key") == payload
    assert fake_redis.ttls["test-key"] == 30

    fake_redis.values[cache.KNOWLEDGE_STATUS_CACHE_KEY] = "{}"
    assert cache.invalidate_knowledge_status_cache() is True
    assert cache.KNOWLEDGE_STATUS_CACHE_KEY not in fake_redis.values


def test_redis_failure_is_a_non_fatal_cache_miss(monkeypatch):
    class BrokenRedis:
        def get(self, _key):
            from redis.exceptions import ConnectionError

            raise ConnectionError("test connection failure")

    monkeypatch.setattr(cache, "redis_client", BrokenRedis())
    monkeypatch.setattr(cache, "REDIS_CACHE_ENABLED", True)

    assert cache.get_cached_json("test-key") is None
    assert cache.check_redis() is False


def test_fixed_window_limit_uses_redis_and_allows_a_cache_outage(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(cache, "redis_client", fake_redis)
    monkeypatch.setattr(cache, "REDIS_CACHE_ENABLED", True)

    assert cache.consume_fixed_window_limit("rate-key", limit=2) is True
    assert cache.consume_fixed_window_limit("rate-key", limit=2) is True
    assert cache.consume_fixed_window_limit("rate-key", limit=2) is False
    assert fake_redis.ttls["rate-key"] == 60
