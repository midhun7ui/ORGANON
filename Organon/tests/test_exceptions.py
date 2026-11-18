from core.exceptions import (
    OrganonError,
    ProviderError,
    RetrievalError,
)

try:
    raise ProviderError("Embedding provider unavailable")

except ProviderError as error:
    print("Provider failed:", error)

except OrganonError as error:
    print("ORGANON error:", error)