"""
Custom exception classes for GuardianRAG
"""


class GuardianRAGException(Exception):
    """Base exception for all GuardianRAG errors"""
    pass


class ConfigurationError(GuardianRAGException):
    """Raised when configuration is invalid or missing"""
    pass


class DocumentProcessingError(GuardianRAGException):
    """Raised when document processing fails"""
    pass


class TokenCountError(GuardianRAGException):
    """Raised when token counting fails"""
    pass


class ValidationError(GuardianRAGException):
    """Raised when validation fails"""
    pass


class APIError(GuardianRAGException):
    """Raised when external API calls fail"""
    pass
