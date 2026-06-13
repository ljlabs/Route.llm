"""
HTTP Client Wrapper

Provides a configured httpx AsyncClient with connection pooling.
"""

import httpx
import logging

logger = logging.getLogger(__name__)


class HTTPClient:
    """
    HTTP client wrapper with connection pooling and reasonable defaults.
    """
    
    def __init__(self, timeout: float = 60.0, connect_timeout: float = 10.0, read_timeout: float = 120.0):
        """
        Initialize HTTP client.
        
        Args:
            timeout: Default timeout for all requests
            connect_timeout: Connection timeout
            read_timeout: Read timeout (higher for streaming)
        """
        self.timeout = httpx.Timeout(
            timeout,
            connect=connect_timeout,
            read=read_timeout
        )
        self._client: httpx.AsyncClient = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create the underlying httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def post(
        self,
        url: str,
        json: dict = None,
        headers: dict = None,
        **kwargs
    ) -> httpx.Response:
        """
        Send POST request.
        
        Args:
            url: Request URL
            json: JSON body
            headers: Request headers
            **kwargs: Additional arguments
            
        Returns:
            Response object
        """
        client = await self.get_client()
        return await client.post(url, json=json, headers=headers, **kwargs)
    
    async def get(
        self,
        url: str,
        headers: dict = None,
        **kwargs
    ) -> httpx.Response:
        """
        Send GET request.
        
        Args:
            url: Request URL
            headers: Request headers
            **kwargs: Additional arguments
            
        Returns:
            Response object
        """
        client = await self.get_client()
        return await client.get(url, headers=headers, **kwargs)
    
    def build_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Request:
        """
        Build a request object (for streaming).
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional arguments
            
        Returns:
            Request object
        """
        return httpx.Request(method, url, **kwargs)
    
    async def send(
        self,
        request: httpx.Request,
        stream: bool = False,
        **kwargs
    ) -> httpx.Response:
        """
        Send a prepared request.
        
        Args:
            request: Prepared request
            stream: Enable streaming
            **kwargs: Additional arguments
            
        Returns:
            Response object
        """
        client = await self.get_client()
        return await client.send(request, stream=stream, **kwargs)
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global instance
_http_client: HTTPClient = None


def get_http_client() -> HTTPClient:
    """Get the global HTTP client instance."""
    global _http_client
    if _http_client is None:
        _http_client = HTTPClient()
    return _http_client


def init_http_client(
    timeout: float = 60.0,
    connect_timeout: float = 10.0,
    read_timeout: float = 120.0
) -> HTTPClient:
    """Initialize the global HTTP client."""
    global _http_client
    _http_client = HTTPClient(timeout, connect_timeout, read_timeout)
    return _http_client