# -*- coding: utf-8 -*-

import argparse
from collections import deque


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', dest='source_file', required=True)
    parser.add_argument('-r', '--rule', dest='rule_file', required=True)

    return parser.parse_args()


def read_rule(rule_file):
    rules = []
    with open(rule_file) as f_in:
        for line in f_in:
            line = line.rstrip()
            if line:
                (src, tgt) = line.split('\t', 2)
                src_tokens = src.split(' ')
                tgt_tokens = tgt.split(' ')
                rules.append((src_tokens, tgt_tokens))
    return rules


def apply_rules(rules, line):
    line_tokens = line.split(' ')
    result_tokens = []
    while line_tokens:
        matched = False
        for rule in rules:
            rule_len = len(rule[0])
            if len(line_tokens) < rule_len:
                continue
            if line_tokens[:rule_len] == rule[0]:
                line_tokens = line_tokens[rule_len:]
                result_tokens.extend(rule[1])
                matched = True
                break
        if not matched:
            result_tokens.append(line_tokens[0])
            line_tokens = line_tokens[1:]
    return ' '.join(result_tokens)


def main(args):
    rules = read_rule(args.rule_file)
    with open(args.source_file) as f_in:
        for line in f_in:
            line = line.rstrip()
            print(apply_rules(rules, line))


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
