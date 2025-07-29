import os
import json
import redis
from typing import Optional, Any


class RedisCache:
    """Redis cache manager for storing and retrieving prediction results"""
    
    def __init__(self):
        self.redis_client = None
        self.enabled = os.environ.get("REDIS_ENABLED", "false").lower() == "true"
        self.host = os.environ.get("REDIS_HOST", "localhost")
        self.port = int(os.environ.get("REDIS_PORT", 6379))
        self.password = os.environ.get("REDIS_PASSWORD", None)
        self.cache_ttl = 86400  # 24 hours
        
        if self.enabled:
            self.initialize()
    
    def initialize(self) -> None:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            print("Redis connection established")
        except Exception as e:
            print(f"Redis connection failed: {str(e)}")
            self.redis_client = None
    
    def is_connected(self) -> bool:
        """Check if Redis connection is active"""
        if not self.enabled or not self.redis_client:
            return False
        try:
            return self.redis_client.ping()
        except:
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.is_connected():
            return None
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Redis get error: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL"""
        if not self.is_connected():
            return False
        try:
            ttl = ttl or self.cache_ttl
            return self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            print(f"Redis set error: {str(e)}")
            return False
