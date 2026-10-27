# Heading-aware chunking, written with only basic Python.
# A chunk never contains text from two different sections.


def get_heading_from_line(line):
    # Returns the line if it is a heading, otherwise None.
    # Markdown headings:  # Title   ## Title   ### Title
    if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
        return line

    # "Chapter 7: Jupiter" style headings
    if line.startswith("Chapter"):
        rest = line[len("Chapter"):].strip()
        digit_count = 0
        while digit_count < len(rest) and rest[digit_count].isdigit():
            digit_count = digit_count + 1
        if digit_count > 0 and digit_count < len(rest) and rest[digit_count] == ":":
            return line

    return None


def split_into_sentences(paragraph):
    # Cuts after . ! or ? but ONLY when a space (or the end) follows,
    # so numbers like 87.97 are not cut in half.
    sentences = []
    current = ""

    for i in range(len(paragraph)):
        char = paragraph[i]
        current = current + char

        if char == "." or char == "!" or char == "?":
            at_end = (i == len(paragraph) - 1)
            next_is_space = False
            if not at_end:
                if paragraph[i + 1] == " " or paragraph[i + 1] == "\n":
                    next_is_space = True

            if at_end or next_is_space:
                sentences.append(current.strip())
                current = ""

    if len(current.strip()) > 0:
        sentences.append(current.strip())

    return sentences


def make_chunk(text, heading):
    chunk = {}
    chunk["text"] = text
    chunk["heading"] = heading
    return chunk


def chunk_text(full_text, chunk_size):
    # ---------- Part 1: cut the document into sections ----------
    lines = full_text.split("\n")

    sections = []
    current_heading = "Introduction / Untitled Section"
    current_has_heading = False
    current_lines = []

    for line in lines:
        found_heading = get_heading_from_line(line.strip())

        if found_heading is not None:
            if len(current_lines) > 0 or current_has_heading:
                section = {}
                section["heading"] = current_heading
                section["has_heading"] = current_has_heading
                section["lines"] = current_lines
                sections.append(section)

            current_heading = found_heading.strip()
            current_has_heading = True
            current_lines = []
        else:
            current_lines.append(line)

    if len(current_lines) > 0 or current_has_heading:
        section = {}
        section["heading"] = current_heading
        section["has_heading"] = current_has_heading
        section["lines"] = current_lines
        sections.append(section)

    # ---------- Part 2: turn each section into chunks ----------
    chunks = []

    for section in sections:
        heading = section["heading"]
        body_text = "\n".join(section["lines"]).strip()

        if section["has_heading"]:
            section_text = (heading + "\n" + body_text).strip()
        else:
            section_text = body_text

        if len(section_text) == 0:
            continue

        # Small section: keep it whole as one chunk.
        if len(section_text) <= chunk_size:
            chunks.append(make_chunk(section_text, heading))
            continue

        # Big section: fill a "bucket" paragraph by paragraph.
        paragraphs = body_text.split("\n\n")
        current = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) == 0:
                continue

            if len(current) + len(paragraph) + 2 <= chunk_size:
                current = current + paragraph + "\n\n"
            else:
                if len(current) > 0:
                    chunks.append(make_chunk("[" + heading + "] " + current.strip(), heading))

                if len(paragraph) > chunk_size:
                    sentences = split_into_sentences(paragraph)
                    current = ""

                    for sentence in sentences:
                        if len(current) + len(sentence) + 1 <= chunk_size:
                            current = current + sentence + " "
                        else:
                            if len(current) > 0:
                                chunks.append(make_chunk("[" + heading + "] " + current.strip(), heading))
                            current = sentence + " "
                else:
                    current = paragraph + "\n\n"

        if len(current) > 0:
            chunks.append(make_chunk("[" + heading + "] " + current.strip(), heading))

    return chunks
