Apply translation rules *greedily* on a file.

```bash
$ python gen_data.py -s source.txt -r rule.txt > target.txt
```

## rule
One rule per line.  A rule consists of a list of source words and a list of target words,
separated by a tab.
```
A B C\ta b
```

## translation
For each line in the source file, from top to down scan through the rules,
and apply the first rule when it matches the first few words of the source line.
