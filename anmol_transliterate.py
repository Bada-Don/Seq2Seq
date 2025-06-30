def transliterate_punjabi(input_list):
    transliteration = []

    char_map = {
        # Vowels
        'ਅ': 'A', 'ਆ': 'Aw', 'ਇ': 'ie', 'ਈ': 'eI', 'ਉ': 'au', 'ਊ': 'aU',
        'ਏ': 'ey', 'ਐ': 'AY', 'ਓ': 'E', 'ਔ': 'AO', 'ਾ': 'w',

        # Lowercase
        'ੳ': 'a', 'ਬ': 'b', 'ਚ': 'c', 'ਦ': 'd', 'ੲ': 'e', 'ਡ': 'f', 'ਗ': 'g',
        'ਹ': 'h', 'ਿ': 'i', 'ਜ': 'j', 'ਕ': 'k', 'ਲ': 'l', 'ਮ': 'm', 'ਨ': 'n',
        'ੋ': 'o', 'ਪ': 'p', 'ਤ': 'q', 'ਰ': 'r', 'ਸ': 's', 'ਟ': 't', 'ੁ': 'u',
        'ਵ': 'v', 'ਣ': 'x', 'ੇ': 'y', 'ਜ਼': 'z',

        # Uppercase
        'ਭ': 'B', 'ਛ': 'C', 'ਧ': 'D', 'ਢ': 'F', 'ਘ': 'G', '੍ਹ': 'H',
        'ੀ': 'I', 'ਝ': 'J', 'ਖ': 'K', 'ਲ਼': 'L', 'ੰ': 'M', 'ਂ': 'N',
        'ੌ': 'O', 'ਫ': 'P', 'ਥ': 'Q', '੍ਰ': 'R', 'ਸ਼': 'S', 'ਠ': 'T',
        'ੂ': 'U', 'ੜ': 'V', 'ਯ': 'X', 'ੈ': 'Y', 'ਗ਼': 'Z',

        # Special
        'ਙ': '|', '◌ੱ': '~', ' ': ' ', 'ੱ': '`'  # include space and backtick explicitly
    }

    i = 0
    while i < len(input_list):
        char = input_list[i]

        # Rule 2: 'ਿ' after '੍ਰ' or '੍ਹ' → place 'i' before i-2th char
        if char == 'ਿ' and i >= 2 and input_list[i - 1] in ['੍ਰ', '੍ਹ']:
            translit_prev2 = char_map.get(input_list[i - 2], input_list[i - 2])
            translit_prev1 = char_map.get(input_list[i - 1], input_list[i - 1])
            translit_i = char_map.get(char, char)
            transliteration = transliteration[:-2]  # remove last two
            transliteration.extend([translit_i, translit_prev2, translit_prev1])
            i += 1

        # Rule 1: Simple 'ਿ' case → place 'i' before i-1th char
        elif char == 'ਿ' and i > 0:
            translit_prev = char_map.get(input_list[i - 1], input_list[i - 1])
            translit_i = char_map.get(char, char)
            transliteration = transliteration[:-1]  # remove last added
            transliteration.extend([translit_i, translit_prev])
            i += 1

        # Rule 3: "੍" and "ਰ" → replace both with 'R'
        elif i < len(input_list) - 1 and char == '੍ਰ' and input_list[i + 1] == 'ਰ':
            transliteration.append('R')
            i += 2  # skip next char too

        # Rule 4: " " between same characters → ` + char
        elif i < len(input_list) - 2 and input_list[i + 1] == '੍' and input_list[i] == input_list[i + 2]:
            translit = char_map.get(input_list[i], input_list[i])
            transliteration.append('`')
            transliteration.append(translit)
            i += 3  # skip current, space, and next repeated char

        # Normal mapping
        else:
            transliteration.append(char_map.get(char, char))
            i += 1

    return transliteration


# Example usage
example = [
  [
    "ਐ",
    "ਮ",
    ".",
    "ਐ",
    "ਲ",
    ".",
    "ਏ"
  ]
]

result = []
for l in example:
    result.append(''.join(transliterate_punjabi(l)))

Res = ' '.join(result)

if __name__ == '__main__':
    print(Res) 


# ------------------------------------# My Prompt #----------------------------------------------- #
# My Prompt
# Follow the following steps:
# 1. Translate the given English word/sentence to Punjabi 
# 2. Make a list of the Punjabi word/sentence like:

# Example 1:
# Input: "Harshit"
# Expected Output:
# [["ਹ", "ਰ", "ਿ", "ਸ਼", "ਤ"]]

# Example 2:
# Input: "Sat Sri Akal"
# Expected Output:
# [
#     ["ਸ", "ਤ"],
#     ["ਸ਼", "੍", "ੀ"],
#     ["ਅ", "ਕ", "ਾ", "ਲ"]
# ]

# Example 3 (testing subscript and matra):
# Input: "prabhat"
# Expected Output:
# [
#     ["ਪ", "੍", "ਭ", "ਾ", "ਤ"]
# ]
