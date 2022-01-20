---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**

A clear and concise description of what the bug is.

**Code for Reproduction**

Paste your code for reproducing the bug:

```r
# your R code
```

or 

```python
# your Python code
```

**Expected behavior**

A clear and concise description of what you expected to happen.

**Desktop (please complete the following information):**
 - OS: [e.g. iOS]
 - R/Python Version [e.g. 3.6.3]
 - Package Version [e.g. 0.3.0]
 
 You can get the information from Python via running:
```python
import platform
print("Platform Version: {0}, {1}".format(platform.platform(), platform.architecture()[0]))
print("Python Version:", platform.python_version())
import abess
print("Package Version:", abess.__version__)
```
And from R via running: 
```r
R.version
packageVersion("abess")
```

**Screenshots**

If needed, add screenshots to help explain your problem.

**Additional context**

Add any other context about the problem here.
