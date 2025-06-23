# http_session_service.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

import httpx
from typing_extensions import override

from .base_session_service import BaseSessionService, GetSessionConfig, ListSessionsResponse
from .session import Session
from ..events.event import Event

# Configure logging following ADK patterns
logger = logging.getLogger('google_adk.' + __name__)

class HttpSessionService(BaseSessionService):
    """
    HTTP-based session service client compatible with Google ADK.
    
    Follows the same patterns as VertexAiSessionService but connects to 
    an HTTP endpoint instead of the Vertex AI API.
    """
    
    def __init__(
        self, 
        service_url: str,
        project: Optional[str] = None,  # For compatibility with VertexAI interface
        location: Optional[str] = None,  # For compatibility with VertexAI interface
        timeout: float = 30.0,
        **kwargs: Any
    ):
        """
        Initialize the HTTP session service client.
        
        Args:
            service_url: Base URL of the session service
            project: Optional project (for interface compatibility)
            location: Optional location (for interface compatibility) 
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to httpx.AsyncClient
        """
        self.service_url = service_url.rstrip('/')
        self.project = project  # Store for compatibility but not used
        self.location = location  # Store for compatibility but not used
        self.timeout = timeout
        self.client_kwargs = kwargs
        self._client: Optional[httpx.AsyncClient] = None
    
    @asynccontextmanager
    async def _get_api_client(self):
        """
        Get HTTP client - following the _get_api_client pattern from VertexAI.
        
        Instantiated per request for proper event loop management,
        similar to the VertexAI implementation.
        """
        client = httpx.AsyncClient(
            timeout=self.timeout,
            **self.client_kwargs
        )
        try:
            yield client
        finally:
            await client.aclose()
    
    def _handle_api_response(self, response: httpx.Response, operation: str, session_id: Optional[str] = None):
        """Handle API response following VertexAI error handling patterns"""
        if response.status_code == 404:
            return None
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get('detail', f'HTTP {response.status_code}')
            except:
                error_msg = f'HTTP {response.status_code}: {response.text}'
            
            logger.error(f"Error {operation} {session_id if session_id else ''}: {error_msg}")
            
            # Re-raise with original exception type for proper error handling
            if response.status_code == 404:
                return None
            elif response.status_code >= 500:
                raise RuntimeError(f"Session service error: {error_msg}")
            else:
                raise ValueError(f"Invalid request: {error_msg}")
        
        response.raise_for_status()
        return response.json()
    
    @override
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """
        Create a new session.
        
        Following the pattern from VertexAiSessionService.create_session()
        """
        try:
            logger.info(f"Creating session for app_name={app_name}, user_id={user_id}")
            
            session_json_dict = {
                'app_name': app_name,
                'user_id': user_id,
                'state': state or {},
                'session_id': session_id
            }
            
            async with self._get_api_client() as api_client:
                response = await api_client.post(
                    f"{self.service_url}/sessions",
                    json=session_json_dict
                )
                
                data = self._handle_api_response(response, "creating session")
                
                # Handle different response formats
                session_data = data if 'app_name' in data else data.get('session', data)
                
                session = Session(
                    app_name=str(session_data['app_name']),
                    user_id=str(session_data['user_id']),
                    id=str(session_data['id']),
                    state=session_data.get('state', {}),
                    last_update_time=session_data.get('last_update_time', 0.0),
                )
                
                # Add events if present
                if 'events' in session_data:
                    session.events = [Event(**event_data) for event_data in session_data['events']]
                
                logger.info(f"Session created successfully: session_id={session.id}")
                return session
                
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

    @override
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """
        Get a specific session by ID.
        
        Following the pattern from VertexAiSessionService.get_session()
        """
        try:
            logger.info(f"Fetching session: session_id={session_id}")
            
            url = f"{self.service_url}/sessions/{app_name}/{user_id}/{session_id}"
            
            # Add query parameters for event filtering if config is provided
            params = {}
            if config:
                if config.num_recent_events is not None:
                    params['num_recent_events'] = config.num_recent_events
                if config.after_timestamp is not None:
                    params['after_timestamp'] = config.after_timestamp
            
            async with self._get_api_client() as api_client:
                response = await api_client.get(url, params=params)
                
                data = self._handle_api_response(response, "getting session", session_id)
                if data is None:
                    logger.info(f"Session not found: session_id={session_id}")
                    return None
                
                # Handle different response formats  
                session_data = data if 'app_name' in data else data.get('session', data)
                
                session = Session(
                    app_name=str(session_data['app_name']),
                    user_id=str(session_data['user_id']),
                    id=str(session_data['id']),
                    state=session_data.get('state', {}),
                    last_update_time=session_data.get('last_update_time', 0.0),
                )
                
                # Add events if present, following VertexAI pattern
                if 'events' in session_data:
                    session.events = [Event(**event_data) for event_data in session_data['events']]
                    # Sort events by timestamp like VertexAI does
                    session.events.sort(key=lambda event: event.timestamp)
                
                logger.info(f"Session retrieved successfully: session_id={session_id}")
                return session
                
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            raise

    @override
    async def list_sessions(
        self, 
        *, 
        app_name: str, 
        user_id: str
    ) -> ListSessionsResponse:
        """
        List all sessions for a user in an app.
        
        Following the pattern from VertexAiSessionService.list_sessions()
        """
        try:
            logger.info(f"Listing sessions for app_name={app_name}, user_id={user_id}")
            
            url = f"{self.service_url}/sessions/{app_name}/{user_id}"
            
            async with self._get_api_client() as api_client:
                response = await api_client.get(url)
                
                data = self._handle_api_response(response, "listing sessions")
                if data is None:
                    return ListSessionsResponse(sessions=[])
                
                sessions_data = data.get('sessions', [])
                sessions = []
                
                for session_data in sessions_data:
                    session = Session(
                        app_name=str(session_data['app_name']),
                        user_id=str(session_data['user_id']),
                        id=str(session_data['id']),
                        state=session_data.get('state', {}),
                        last_update_time=session_data.get('last_update_time', 0.0),
                    )
                    
                    # Add events if present
                    if 'events' in session_data:
                        session.events = [Event(**event_data) for event_data in session_data['events']]
                    
                    sessions.append(session)
                
                logger.info(f"Retrieved {len(sessions)} sessions for app_name={app_name}, user_id={user_id}")
                return ListSessionsResponse(sessions=sessions)
                
        except Exception as e:
            logger.error(f"Failed to list sessions for app_name={app_name}, user_id={user_id}: {e}")
            raise

    @override
    async def delete_session(
        self, 
        *, 
        app_name: str, 
        user_id: str, 
        session_id: str
    ) -> None:
        """
        Delete a specific session.
        
        Following the pattern from VertexAiSessionService.delete_session()
        """
        try:
            logger.info(f"Deleting session: session_id={session_id}")
            
            url = f"{self.service_url}/sessions/{app_name}/{user_id}/{session_id}"
            
            async with self._get_api_client() as api_client:
                response = await api_client.delete(url)
                
                self._handle_api_response(response, "deleting session", session_id)
                logger.info(f"Session deleted successfully: session_id={session_id}")
                
        except Exception as e:
            logger.error(f'Error deleting session {session_id}: {e}')
            raise

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        """
        Append an event to a session.
        
        Following the pattern from VertexAiSessionService.append_event()
        First updates the in-memory session, then sends to API.
        """
        try:
            logger.info(f"Appending event to session_id={session.id}")
            
            # Update the in-memory session first (like VertexAI does)
            await super().append_event(session=session, event=event)
            
            # Send to API using the VertexAI-style endpoint
            url = f"{self.service_url}/sessions/{session.app_name}/{session.user_id}/{session.id}/appendEvent"
            payload = {"event": event.model_dump(mode="json")}
            
            async with self._get_api_client() as api_client:
                response = await api_client.post(url, json=payload)
                
                data = self._handle_api_response(response, "appending event", session.id)
                
                # Return the event (the API might have updated it)
                if data and 'event' in data:
                    return Event(**data['event'])
                
                logger.info(f"Event appended successfully to session_id={session.id}")
                return event
                
        except Exception as e:
            logger.error(f"Failed to append event to session {session.id}: {e}")
            raise