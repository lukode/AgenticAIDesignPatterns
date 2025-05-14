import json
import re
import logging

QUERY_TAG = "<question>"
QUERY_TAG_END = "</question>"
THOUGHT_TAG = "<thought>"
THOUGHT_TAG_END = "</thought>"
OBSERVATION_TAG = "<observation>"
OBSERVATION_TAG_END = "</observation>"
RESPONSE_TAG = "<answer>"
RESPONSE_TAG_END = "</answer>"


def sanitize_json_string(json_str: str) -> str:
    """
    Sanitizes a JSON string to ensure it can be properly parsed by json.loads().
    Handles common issues like unescaped quotes, unnecessary escapes, etc.

    Args:
        json_str: The JSON string to sanitize

    Returns:
        A sanitized JSON string that should be parseable by json.loads()
    """
    # Skip sanitization if the string is already valid JSON
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    # Make a copy of the original for logging
    original = json_str

    try:
        # Step 1: Try to identify and fix the most common JSON structure issues

        # Replace triple backslashes with single backslashes
        json_str = re.sub(r"\\\\\\", r"\\", json_str)

        # Replace double backslashes with single backslashes
        json_str = re.sub(r"\\\\", r"\\", json_str)

        # Remove trailing commas in objects and arrays
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # Ensure property names are properly quoted
        json_str = re.sub(r"([{,]\s*)([a-zA-Z0-9_]+)(\s*:)", r'\1"\2"\3', json_str)

        # Step 2: Advanced string content handling

        # Parse the JSON structure to identify string values
        # This is a more robust approach than simple character-by-character scanning

        # First, try to identify the overall structure using regex
        # Look for patterns like "key": "value" or "key": {...}
        string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)":\s*"([^"\\]*(\\.[^"\\]*)*)"'

        # Find all string values and process them
        def escape_string_content(match):
            key = match.group(1)
            value = match.group(3)

            # Escape single quotes in the value
            value = value.replace("'", "\\'")

            # Escape unescaped double quotes in the value
            value = re.sub(r'(?<!\\)"', '\\"', value)

            return f'"{key}": "{value}"'

        # Apply the string content escaping
        json_str = re.sub(string_pattern, escape_string_content, json_str)

        # Step 3: Handle nested content tags and special XML-like structures
        # This is specifically for the case in the error example with <content>...</content> tags

        # Find content between XML-like tags and escape it properly
        tag_pattern = r"(<[a-zA-Z]+>)(.*?)(</[a-zA-Z]+>)"

        def escape_tag_content(match):
            opening_tag = match.group(1)
            content = match.group(2)
            closing_tag = match.group(3)

            # Escape quotes and backslashes in the content
            content = content.replace("\\", "\\\\")
            content = content.replace('"', '\\"')
            content = content.replace("'", "\\'")

            return f"{opening_tag}{content}{closing_tag}"

        json_str = re.sub(tag_pattern, escape_tag_content, json_str, flags=re.DOTALL)

        # Step 4: Character-by-character processing for any remaining issues
        # This is a fallback for complex cases that the regex patterns might miss

        # Process the string character by character to handle nested quotes
        in_string = False
        escaped = False
        chars = list(json_str)

        for i in range(len(chars)):
            if chars[i] == "\\":
                escaped = not escaped
            elif chars[i] == '"' and not escaped:
                in_string = not in_string
            elif chars[i] == "'" and in_string and not escaped:
                # Replace unescaped single quotes with escaped single quotes inside JSON strings
                chars[i] = "\\'"
            elif chars[i] == '"' and in_string and not escaped:
                # This is an unescaped double quote inside a string
                chars[i - 1] = chars[i - 1] + "\\"
            else:
                escaped = False

        json_str = "".join(chars)

        # Try to parse the sanitized JSON
        try:
            json.loads(json_str)
            if json_str != original:
                logging.debug("JSON sanitized successfully")
            return json_str
        except json.JSONDecodeError as e:
            # If standard sanitization fails, try a more aggressive approach
            logging.warning(
                f"Standard sanitization failed: {e}. Trying aggressive sanitization."
            )

            # Last resort: Try to extract and rebuild the JSON structure
            try:
                # Extract the basic structure: function name and arguments
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_str)

                if name_match:
                    func_name = name_match.group(1)

                    # Extract the arguments section
                    args_match = re.search(r'"arguments"\s*:\s*({.+})', json_str)
                    if args_match:
                        args_str = args_match.group(1)

                        # Extract key-value pairs from arguments
                        key_value_pairs = {}
                        kv_pattern = r'"([^"]+)"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
                        for kv_match in re.finditer(kv_pattern, args_str):
                            key = kv_match.group(1)
                            value = kv_match.group(2)
                            # Clean the value
                            value = value.replace("'", "\\'")
                            key_value_pairs[key] = value

                        # Rebuild the JSON
                        args_json = json.dumps(key_value_pairs)
                        rebuilt_json = (
                            f'{{"name": "{func_name}", "arguments": {args_json}}}'
                        )

                        # Verify it's valid JSON
                        json.loads(rebuilt_json)
                        logging.info("Successfully rebuilt JSON structure")
                        return rebuilt_json
            except Exception as rebuild_error:
                logging.warning(f"Failed to rebuild JSON: {rebuild_error}")

            # If all else fails, return a simplified valid JSON with error info
            error_json = json.dumps(
                {
                    "name": "error",
                    "arguments": {
                        "error": f"Failed to parse JSON: {str(e)}",
                        "original": original[:100] + "..."
                        if len(original) > 100
                        else original,
                    },
                }
            )
            return error_json

    except Exception as e:
        # Catch any unexpected errors in the sanitization process
        logging.error(f"Unexpected error in JSON sanitization: {e}")

        # Create a valid JSON with error information
        error_json = json.dumps(
            {
                "name": "error",
                "arguments": {
                    "error": f"Sanitization error: {str(e)}",
                    "original": original[:100] + "..."
                    if len(original) > 100
                    else original,
                },
            }
        )

        return error_json
