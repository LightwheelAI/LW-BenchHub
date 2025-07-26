def convert_fixture_to_name(d) -> dict:
    if not isinstance(d, dict):
        # Check if it is a fixture type
        if hasattr(d, "__class__") and "lwlab.core.models.fixtures" in d.__class__.__module__:
            return d.name
        return d
    result = {}
    for k, v in d.items():
        result[k] = convert_fixture_to_name(v)
    return result
