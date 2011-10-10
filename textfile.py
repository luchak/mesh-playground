#!/usr/bin/env python

def NonCommentLineTokens(filename):
  with open(filename, 'r') as input:
    for line in input:
      line = line.strip()
      if line != '' and line[0] != '#':
        yield line.split()

def ParseLineTokens(line_tokens, line_parse_fn,num_lines=None):
  if num_lines is None:
    return [line_parse_fn(tokens) for tokens in line_tokens]
  else:
    return [line_parse_fn(line_tokens.next()) for i in xrange(num_lines)]

