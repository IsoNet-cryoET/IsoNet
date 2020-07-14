from __future__ import absolute_import
from __future__ import print_function
import re
import six


class RuleError(RuntimeError):
    def __init__(self, rule):
        RuntimeError.__init__(self)
        self.Rule = rule

    def __str__(self):
        return repr(self.Rule)


class InvalidTokenError(RuntimeError):
    def __init__(self, s, str_index):
        RuntimeError.__init__(self)
        self.string = s
        self.index = str_index

    def __str__(self):
        return "at index %d: ...%s..." % \
               (self.index, self.string[self.index:self.index+20])


class ParseError(RuntimeError):
    def __init__(self, msg, s, token):
        RuntimeError.__init__(self)
        self.message = msg
        self.string = s
        self.Token = token

    def __str__(self):
        if self.Token is None:
            return "%s at end of input" % self.message
        else:
            return "%s at index %d: ...%s..." % \
                   (self.message, self.Token[2],
                           self.string[self.Token[2]:self.Token[2]+20])


class RE(object):
    def __init__(self, s, flags=0):
        self.Content = s
        self.RE = re.compile(s, flags)

    def __repr__(self):
        return "RE(%s)" % self.Content


def _matches_rule(rule, s, start, rule_dict, debug=False):
    if debug:
        print("Trying", rule, "on", s[start:])

    if isinstance(rule, tuple):
        if rule[0] == "|":
            for subrule in rule[1:]:
                length, match_obj = _matches_rule(
                        subrule, s, start, rule_dict, debug)
                if not length:
                    continue
                return length, match_obj
        else:
            my_match_length = 0
            for subrule in rule:
                length, _ = _matches_rule(
                        subrule, s, start, rule_dict, debug)
                if not length:
                    break
                my_match_length += length
                start += length
            else:
                return my_match_length, None
        return 0, None

    elif isinstance(rule, six.string_types):
        return _matches_rule(rule_dict[rule], s, start, rule_dict, debug)

    elif isinstance(rule, RE):
        match_obj = rule.RE.match(s, start)
        if match_obj:
            return match_obj.end()-start, match_obj
        return 0, None

    raise RuleError(rule)


def lex(lex_table, s, debug=False, match_objects=False):
    rule_dict = dict(lex_table)
    result = []
    i = 0
    while i < len(s):
        for name, rule in lex_table:
            length, match_obj = _matches_rule(rule, s, i, rule_dict, debug)
            if length:
                if match_objects:
                    result.append((name, s[i:i+length], i, match_obj))
                else:
                    result.append((name, s[i:i+length], i))
                i += length
                break
        else:
            raise InvalidTokenError(s, i)
    return result


class LexIterator(object):
    def __init__(self, lexed, raw_str, lex_index=0):
        self.lexed = lexed
        self.raw_string = raw_str
        self.index = lex_index

    def copy(self):
        return type(self)(self.lexed, self.raw_string, self.index)

    def assign(self, rhs):
        assert self.lexed is rhs.lexed
        assert self.raw_string is rhs.raw_string

        self.index = rhs.index

    def next_tag(self, i=0):
        return self.lexed[self.index + i][0]

    def next_str(self, i=0):
        return self.lexed[self.index + i][1]

    def next_match_obj(self):
        return self.lexed[self.index][3]

    def next_str_and_advance(self):
        result = self.next_str()
        self.advance()
        return result

    def advance(self):
        self.index += 1

    def is_at_end(self, i=0):
        return self.index + i >= len(self.lexed)

    def is_next(self, tag, i=0):
        return (
                self.index + i < len(self.lexed)
                and self.next_tag(i) is tag)

    def raise_parse_error(self, msg):
        if self.is_at_end():
            raise ParseError(msg, self.raw_string, None)

        raise ParseError(msg, self.raw_string, self.lexed[self.index])

    def expected(self, what_expected):
        if self.is_at_end():
            self.raise_parse_error(
                    "%s expected, end of input found instead" %
                    what_expected)
        else:
            self.raise_parse_error(
                    "%s expected, %s found instead" %
                    (what_expected, str(self.next_tag())))

    def expect_not_end(self):
        if self.is_at_end():
            self.raise_parse_error("unexpected end of input")

    def expect(self, tag):
        self.expect_not_end()
        if not self.is_next(tag):
            self.expected(str(tag))
