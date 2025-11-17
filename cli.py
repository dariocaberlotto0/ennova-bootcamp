"""Using `argparse`, write a command-line utility that "normalizes" its (string) input, namely:

1. Converts to lowercase
1. Strips punctuation unless requested not to (with `--no-strip-punct`)
1. Drops words with length below the `--min-len` arg (which defaults to 1)

The following cell will write your code to the `cli.py` file and show the results of its execution.
"""

import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description='Process some strings.')
    parser.add_argument('text', type=str, help='Input text to process', nargs=argparse.REMAINDER)
    parser.add_argument('--min-len', type=int, default=1, help='Minimum length of text to display')
    parser.add_argument('--no-strip-punct', action='store_true', help='Do not strip punctuation from the text')
    args = parser.parse_args(argv)

    text = ' '.join(args.text).lower()
    if not args.no_strip_punct:
        import string
        # Remove punctuation
        # text = ''.join(char for char in text if char not in string.punctuation)
        text = ''.join(filter(lambda x: x not in string.punctuation, text))
    
    words = text.split()
    # Filter words by minimum length
    # normalized_words = [word for word in words if len(word) >= args.min_len]
    normalized_words = filter(lambda x: len(x) >= args.min_len, words)
    result = ' '.join(normalized_words)
    print(result)

if __name__ == '__main__':
    main()

# python cli.py --min-len 2 'Hello, Bootcamp!'                      Output: "hello bootcamp"
# python cli.py --min-len 2 --no-strip-punct 'Hello, Bootcamp!'     Output: "hello, bootcamp!"