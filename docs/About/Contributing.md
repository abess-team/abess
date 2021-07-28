# Contributing

Contributions are welcome. No matter your current skills, it’s possible to make valuable contribution to the abess.

## Bugs Report 

If you’ve found a bug, please open an issue at [https://github.com/abess-team/abess/issues](https://github.com/abess-team/abess/issues) or send an email to Jin Zhu at zhuj37@mail2.sysu.edu.cn. 
When reporting a bug, please include:     
- codes to reproduce the bug.
- your operating system and Python or R version.
- any details about your local setup that might be helpful in troubleshooting. 

We strongly encourage to spend some time trying to make it as minimal as possible: the more time you spend doing this, the easier it will be for the abess team to fix it.  

## Suggest New Features

If you’re working on best subset selection for some problem that can not be handled by the abess library, 
it is encouraged to share your new features suggestion to us. 
You can open an issue at [https://github.com/abess-team/abess/issues](https://github.com/abess-team/abess/issues) to post your suggestion.
When suggesting a new feature, please:
- explain in detail how it would work.
- keep the scope as narrow as possible, to make it easier to understand and implementation.
- provide few important literatures if possible.

## Contribute documentation
If you’re a more experienced with the abess and are looking forward to improve your open source development skills, the next step up is to contribute a pull request to a abess library. 

In most of case, the workflow is: 
- fork github repository [abess](https://github.com/abess-team/abess);
- check out a branch from your repository;
- commit your improvements or additions to documentation (forming an ideally legible commit history);
- merge your branch with the main branch that have the up-to-date codes in [abess](https://github.com/abess-team/abess);
- submit a pull request explaining your contribution for documentation.

For the development of R documentation, the most important thing to know is that the abess R package relies on [roxygen2](https://cran.r-project.org/web/packages/roxygen2) package. This means that documentation is found in the R code close to the source of each function. 
Before writing the documentation, it would be better to ensure the usage of the [Rd tags](https://cran.r-project.org/web/packages/roxygen2/vignettes/rd.html). 

For the development of Python documentation, there is a little difference between a new method and a new function. A new method need a brief introduction and some examples, such as [[link]](https://github.com/abess-team/abess/blob/master/python/abess/linear.py#:~:text=class%20abessLogistic(bess_base)%3A-,%22%22%22,%22%22%22,-def%20__init__(self)); and a new function under should at least contain an introduction and the parameters it requires, such as [[link]](https://github.com/abess-team/abess/blob/master/python/abess/linear.py#:~:text=return%20y-,def%20score(self%2C%20X%2C%20y)%3A,%22%22%22,-X%2C%20y%20%3D%20self). 
Also note that the style of Python document is similar to [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).

## Contribute code      
If you are a experienced programmer, you might want to help new features development or bug fixing for the abess library. 
Before programming, you should always open an issue and make sure someone from the abess team agrees that your work is really contributive, and is happy with your proposal. We don’t want you to spend a bunch of time on something that we are working on or we don’t think is a good idea.

Also make sure to read the abess style guide (**TODO**) which will make sure that your new code and documentation matches the existing style. This makes the review process much smoother. For more details about code developing, read the [Code Developing](CodeDeveloping.md) description for abess library.