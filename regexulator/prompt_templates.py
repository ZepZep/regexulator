from dataclasses import dataclass, field, asdict
    
FINAL_REGEX = """Make sure all parentheses match.
Write the final regex on the last line with a heading of FINAL REGEX: ... Make sure it is on the same line as the heading and is not inside a code block.
Do not write anything else after that."""

# arguments: examples, extractions, ?cheatsheet, ?description
START = rf"""{{cheatsheet}}Write a regular expression that matches all the following parts of text wrapped in \` and \` as close as possible (but does not have to be exact).
    The \` and `\ only highlight the desired extraction in the examples, they do not appear in the original text and must not be part of the final regex.
{{description}}
## Examples with added \` \` highlights:
{{examples}}


## Only extractions:
{{extractions}}

First, write down common patterns in the extraction. Use them to construct the regex. Mind the order of the patterns.

{FINAL_REGEX}
"""

# arguments: pattern, examples_tp, extractions_tp, examples_fn, extractions_fn, examples_fp, extractions_fp, ?cheatsheet, ?description
IMPROVE = rf"""{{cheatsheet}}Improve the following regular expression to better match the following positive examples but not match the negative examples.
{{description}}
## Regular expression to improve:
{{pattern}}

## Positive examples that the expression already matches and should remain to match (highlighted with added \` \`):
{{examples_tp}}

### Positive extractions only (expression already matches and should remain to match):
{{extractions_tp}}


## Additional positive examples that the regex should also match (highlighted with added \` \`):
{{examples_fn}}

### Extractions only (expression should also match):
{{extractions_fn}}


## Negative examples (do *NOT* match, do not match, don't match) (highlighted with added \` \`):
{{examples_fp}}

## Negative extractions only (do *NOT* match, do not match, don't match):
{{extractions_fp}}


First, write down how to differentiate between the positive and negative examples and extractions and find common patterns in them. Use this to construct the regex. Mind the order of the patterns. Make sure to match the positive examples but not match the negative examples.

{FINAL_REGEX}
"""
#  Take the context into account and if necessary, include positive (?=) or negative (?!) lookaheads.

NO_FINAL_REGEX = f"""I could not find the final regex.
{FINAL_REGEX}
"""

SELF_REVISE_QUESTION = "Is this correct? Answer only yes or no."
SELF_REVISE_FIX = f"""This does not seem to be correct. Write the what is wrong and how to improve it.
{FINAL_REGEX}
"""

# arguments: error
NOT_COMPILABLE = f"""This regular expression is cannot be compiled. It gives the following error:
{{error}}

Explain what is wrong and improve it.
{FINAL_REGEX}
"""


REGEXR_CHEATSHEET = r"""## Regex Cheat sheet:
Character classes
.	any character except newline
\w\d\s	word, digit, whitespace
\W\D\S	not word, digit, whitespace
[abc]	any of a, b, or c
[^abc]	not a, b, or c
[a-g]	character between a & g
Anchors
^abc$	start / end of the string
\b\B	word, not-word boundary
Escaped characters
\.\*\\	escaped special characters
\t\n\r	tab, linefeed, carriage return
Groups & Lookaround
(abc)	capture group
\1	backreference to group #1
(?:abc)	non-capturing group
(?=abc)	positive lookahead
(?!abc)	negative lookahead
Quantifiers & Alternation
a*a+a?	0 or more, 1 or more, 0 or 1
a{5}a{2,}	exactly five, two or more
a{1,3}	between one & three
a+?a{2,}?	match as few as possible
ab|cd	match ab or cd

## TASK"""


@dataclass(kw_only=True, frozen=True)
class PromptTemplates:
    # arguments: examples, extractions, ?cheatsheet, ?description
    start: str = START
    
    # arguments: pattern, examples_tp, extractions_tp, examples_fn,
    # extractions_fn, examples_fp, extractions_fp
    improve: str = IMPROVE
    
    no_final_regex: str = NO_FINAL_REGEX
    
    self_revise_question: str = SELF_REVISE_QUESTION
    
    self_revise_fix: str = SELF_REVISE_FIX
    
    # arguments: error
    not_compilable: str = NOT_COMPILABLE
    
    cheatsheet: str = REGEXR_CHEATSHEET

    def __str__(self):
        return self.__class__.__name__+"()"

    def __repr__(self):
        return self.__class__.__name__+"()"


# prompt_templates = {
#     "start": START, # arguments: examples, extractions, cheatsheet
#     "improve": IMPROVE, # arguments: pattern, examples_tp, extractions_tp, examples_fn,
#                         # extractions_fn, examples_fp, extractions_fp
#     "no_final_regex": NO_FINAL_REGEX,
#     "self_revise_question": SELF_REVISE_QUESTION, 
#     "self_revise_fix": SELF_REVISE_FIX,
#     "not_compilable": NOT_COMPILABLE, # arguments: error
#     "cheatsheet": REGEXR_CHEATSHEET,
# }
