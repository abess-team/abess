## Comment

`abess` is remove from CRAN because incorrect C++ flags in configure scripts. This submission has address this incorrectness.

## Test environments
* local R installation, R 4.1.0
* win-builder (devel)
* rhub

## R CMD check results

* local R installation, R 4.1.0: 0 errors | 0 warnings | 0 note
* win-builder: 0 errors | 0 warnings | 1 note
* rhub: notes about doi.

1 note in win-builder:

```
Possibly misspelled words in DESCRIPTION:
  multinomial (22:584)
  
CRAN repository db overrides:
  X-CRAN-Comment: Archived on 2023-02-18 as configure issues were not
    corrected in time.

  CXX not set correctly.
```

First, the word "multinomial" is not misspelled. As we have checked in Google, this word is widely used. The typical examples include multinomial distribution and multinomial logistic regression.

Second, the CXX is correct in the submission. The check under the win-builder haven't report this issue.  

### Response NOTEs reported by rhub

```
Found the following (possibly) invalid URLs:
  URL: https://doi.org/10.1002/cem.3289
    From: man/generate.spc.matrix.Rd
    Status: 503
    Message: Service Unavailable
  URL: https://doi.org/10.1073/pnas.2014241117
    From: man/abess.Rd
          man/abesspca.Rd
          man/abessrpca.Rd
    Status: 503
    Message: Service Unavailable
  URL: https://doi.org/10.1111/j.1467-9868.2008.00674.x
    From: man/abess.Rd
    Status: 503
    Message: Service Unavailable

Found the following (possibly) invalid DOIs:
  DOI: 10.1073/pnas.2014241117
    From: DESCRIPTION
    Status: Service Unavailable
    Message: 503
  DOI: 10.1111/j.1467-9868.2008.00674.x
    From: DESCRIPTION
    Status: Service Unavailable
    Message: 503
```

These websites are accessible, and dois are valid. I have checked the availability of them on multiple local computers.

### response to one addtional NOTE in R CRAN

R CRAN also reports the following NOTE:

```
Result: NOTE
     installed size is 96.4Mb
     sub-directories of 1Mb or more:
     libs 95.4Mb
```

We believe this NOTE cannot be fixed at present. 
This NOTE occurs because our sub-directory includes a C++ source code 
provided in https://github.com/yixuan/spectra/, 
but the latest C++ code haven't been published in R CRAN. 
So, we hope you can mercifully accept this release.
