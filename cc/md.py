import re


def extract_heading_content(markdown_text: str, heading_name: str) -> None | str:
    """
    Finds the first heading with the given name at any level and extracts all
    content underneath it, stopping at the next heading of the same or higher
    level. It correctly ignores any headers within fenced code blocks.

    Args:
        markdown_text: The full markdown string.
        heading_name: The text of the heading to find (e.g., "Introduction").

    Returns:
        The content under the specified heading, or None if not found.
    """
    lines = markdown_text.split("\n")
    content_lines = []
    in_code_block = False
    found_heading_level = 0  # Level of the heading we are looking for (1-6)

    for line in lines:
        # Toggle code block state but continue collecting lines if we're in the target section
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            if found_heading_level:
                content_lines.append(line)
            continue

        # If we are inside a code block, just collect the line and skip heading checks
        if in_code_block:
            if found_heading_level:
                content_lines.append(line)
            continue

        # Check for a heading using regex
        match = re.match(r"^(#+)\s+(.*)", line)
        if match:
            current_level = len(match.group(1))
            current_name = match.group(2).strip()

            if not found_heading_level and current_name == heading_name:
                # Found the start of our target section
                found_heading_level = current_level
                continue

            if found_heading_level and current_level <= found_heading_level:
                # Found the end of our section (next heading of same or higher level)
                break

        # Collect content if we are within the target section
        if found_heading_level:
            content_lines.append(line)

    return "\n".join(content_lines).rstrip() if found_heading_level else None


def extract_code_block(text: str) -> None | str:
    """
    Extract the first code block from the given text.

    Args:
        text: The text containing a potential code block

    Returns:
        The code block content (without the backticks), or None if not found
    """
    if not text:
        return None

    # Pattern explanation:
    # ^\s*```[^`]*?$ - Opening line: exactly 3 backticks, non-backtick chars, end of line
    # (.*?) - Capture the code content (non-greedy)
    # ^\s*```\s*$ - Closing line: optional whitespace, exactly 3 backticks, optional whitespace, end of line
    pattern = r"^\s*```[^`]*?$\n(.*?)^\s*```\s*$"

    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).rstrip("\n") if match else None
