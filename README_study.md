# Study the SGLang code

## Separate study material from main branch

All material for studying SGLang are checked into the `study` branch, are under two directories:

- study: Any static files;
- studyIterations: Multiple trials to run the SGLang on `macOS`, separated into directories with date suffix. 

## New iteration

We created a base branch `iter_base`, which contains all things in `study` except the pass trial directories;
Create a new branch iter<YYMMDD> from `iter_base`, try things, then merge the new branch iter<YYMMDD> back into `study`.

```shell
# from the repo root
git checkout iter_base
git pull
git checkout -b iter<YYMMDD>

# from the repo root
cd studyIterations
mkdir iter<YYMMDD>
cd iter<YYMMDD>

# create file like Prompt_newIdeasXXX.md
# do your work and tests;
# Then commit
git commit "Save iter<YYMMDD>"

# Then merge branch iter<YYMMDD> into study branch
git checkout study
git merge iter<YYMMDD>
```