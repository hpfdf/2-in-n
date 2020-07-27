# The 2-in-N Problem

Given N bottles of samples, up to 2 of them are poisoned.

You can use mice to exam any subsets of samples.  For each exam it will tell whether or not the subset contains poisoned sample.

Design a strategy that minimize the number of exams in the worst case.  Also consider strategies when interaction is limited.

## Current Bests

### Interactive
```
N = 1000
Exams = 20    <= 2*log_2(N)
Theoretical Bound: 19     log_2(N*(N+1)/2+1)
```

### 2-Round
```
N = 1000
Exams = 26   < 3*log_2(N)
```

### 1-Round
```
N = 1000
Exams = 47   log_2(N)^log_2(3)
Exams = 40.  4*log_2(N) using BCH code, unimplemented yet.
```
