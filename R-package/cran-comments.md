## Comment

A new release. We update info about citation. 

## Test environments
* local R installation, R 4.1.0
* win-builder (devel)

## R CMD check results

* local R installation, R 4.1.0: 0 errors | 0 warnings | 0 note
* win-builder: 0 errors | 0 warnings | 1 note

1 note in win-builder:

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
  URL: https://www.pnas.org/doi/10.1073/pnas.2014241117
    From: inst/CITATION
          inst/doc/v01-abess-guide.html
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

### response to NOTE in R CRAN

Last released version's CRAN status: OK: 7, NOTE: 6
(See: <https://CRAN.R-project.org/web/checks/check_results_abess.html>), 
where the NOTE reports:
```
Result: NOTE 
     installed size is 70.2Mb
     sub-directories of 1Mb or more:
     libs 69.2Mb 
```

We believe this note cannot be fixed at present. 
This NOTE occurs because our sub-directory includes a C++ source code 
provided in https://github.com/yixuan/spectra/, 
but the latest C++ code haven't been published in R CRAN. 
So, we hope you can mercifully accept this release.
