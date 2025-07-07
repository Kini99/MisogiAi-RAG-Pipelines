# API Authentication Guide

## Overview

This document provides comprehensive guidance on implementing secure authentication for our REST API endpoints.

## Authentication Methods

### 1. API Key Authentication

The simplest form of authentication uses API keys.

```python
import requests

headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

response = requests.get('https://api.example.com/users', headers=headers)
```

### 2. OAuth 2.0 Authentication

For more secure applications, we support OAuth 2.0.

#### Authorization Code Flow

1. Redirect users to the authorization endpoint
2. Handle the authorization callback
3. Exchange the authorization code for an access token

```python
# Step 1: Redirect to authorization endpoint
auth_url = "https://api.example.com/oauth/authorize"
params = {
    'client_id': 'YOUR_CLIENT_ID',
    'redirect_uri': 'https://your-app.com/callback',
    'response_type': 'code',
    'scope': 'read write'
}
```

#### Client Credentials Flow

For server-to-server communication:

```python
import requests

token_url = "https://api.example.com/oauth/token"
data = {
    'grant_type': 'client_credentials',
    'client_id': 'YOUR_CLIENT_ID',
    'client_secret': 'YOUR_CLIENT_SECRET'
}

response = requests.post(token_url, data=data)
access_token = response.json()['access_token']
```

## Security Best Practices

### 1. Token Management

- Store tokens securely
- Implement token refresh logic
- Set appropriate token expiration times

### 2. Rate Limiting

Our API implements rate limiting to prevent abuse:

- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated users

### 3. HTTPS Only

All API communications must use HTTPS. HTTP requests will be rejected.

## Error Handling

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 401 | Unauthorized | Check your authentication credentials |
| 403 | Forbidden | Verify your API key has the required permissions |
| 429 | Too Many Requests | Implement exponential backoff |

### Example Error Response

```json
{
  "error": "invalid_token",
  "error_description": "The access token is invalid or expired",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## SDK Examples

### Python SDK

```python
from example_sdk import ExampleAPI

# Initialize with API key
api = ExampleAPI(api_key='YOUR_API_KEY')

# Make authenticated requests
users = api.users.list()
user = api.users.get(user_id=123)
```

### JavaScript SDK

```javascript
const ExampleAPI = require('example-sdk');

const api = new ExampleAPI({
  apiKey: 'YOUR_API_KEY'
});

// Make authenticated requests
const users = await api.users.list();
const user = await api.users.get(123);
```

## Testing

### Test Your Integration

Use our sandbox environment for testing:

```bash
# Set sandbox environment
export API_BASE_URL="https://sandbox-api.example.com"

# Test authentication
curl -H "Authorization: Bearer YOUR_SANDBOX_API_KEY" \
     https://sandbox-api.example.com/users
```

### Common Issues

1. **Invalid API Key**: Ensure your API key is correct and active
2. **Expired Token**: Implement token refresh logic
3. **Rate Limit Exceeded**: Add retry logic with exponential backoff

## Support

For authentication-related issues:

- Check our [API Status Page](https://status.example.com)
- Review the [API Documentation](https://docs.example.com)
- Contact support at support@example.com

## Changelog

### Version 2.1.0 (2024-01-15)
- Added support for OAuth 2.0 PKCE flow
- Improved error messages
- Enhanced rate limiting

### Version 2.0.0 (2023-12-01)
- Migrated to OAuth 2.0
- Deprecated API key authentication for sensitive endpoints
- Added multi-factor authentication support 