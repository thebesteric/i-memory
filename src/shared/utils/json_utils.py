import json
from typing import Any


def coerce_json_field(value: Any, default: Any) -> Any:
	"""Return parsed JSON for text payloads, otherwise pass through with a safe default."""
	if value is None:
		return default
	if isinstance(value, (str, bytes, bytearray)):
		try:
			parsed = json.loads(value)
		except (TypeError, json.JSONDecodeError):
			return default
		return default if parsed is None else parsed
	return value

