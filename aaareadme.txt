usage: pt2.py [-h] [-s STRING] [-g N] grxml_file

ParseTool: Validates a .grxml file, parses strings, or generates sentences.

positional arguments:
  grxml_file            Path to the .grxml speech recognition grammar file.

options:
  -h, --help            show this help message and exit
  -s STRING, --string STRING
                        A quoted string to parse against the grammar.
  -g N, --generate N    Generate N sentences from the grammar.
  

# successful parse of covered string
python pt2.py  test_hello_world.grxml -s 'hello world'
Successfully validated GRXML structure: test_hello_world.grxml
Attempting to parse string: 'hello world'
True

# successful rejection of non-covered string
python pt2.py  test_hello_world.grxml -s 'hello' 
Successfully validated GRXML structure: test_hello_world.grxml
Attempting to parse string: 'hello'
False

# generation of sentences from grammar
python pt2.py  test_hello_world.grxml -g 3
Successfully validated GRXML structure: test_hello_world.grxml
Attempting to generate 3 sentences...
hello world
hello world
hello world

# parse of sequences
python pt2.py  test_sequence.grxml -s 'play music' 
Successfully validated GRXML structure: test_sequence.grxml
Attempting to parse string: 'play music'
True

# generation from sequences
python pt2.py  test_sequence.grxml -g 10
Successfully validated GRXML structure: test_sequence.grxml
Attempting to generate 10 sentences...
pause music
play music
pause music
play music
pause music now
play music
pause music now
pause music
pause music now
play music

# parse of one-of
python pt2.py  test_one_of.grxml -s 'start please'
Successfully validated GRXML structure: test_one_of.grxml
Attempting to parse string: 'start please'
True

