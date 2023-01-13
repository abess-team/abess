Contribute Python/R code
========================

If you are a experienced programmer, you might want to help new features
development or bug fixing for the abess library. The preferred workflow
for contributing code to abess is to fork the master repository on
GitHub, clone, and develop on a branch:

1. Before contributing, you should always open an issue and make sure
   someone from the abess team agrees that your work is really
   contributive, and is happy with your proposal. We don’t want you to
   spend a bunch of time on something that we are working on or we don’t
   think is a good idea;

2. Fork the `master repository <https://github.com/abess-team/abess>`__
   by clicking on the “Fork” button on the top right of the page, which
   would create a copy to your own GitHub account. If you have forked abess
   repository, enter it and click "Fetch upsteam", followed by "Fetch and merge"
   to ensure the code is the latest one;

3. Clone your fork of abess to the local by
   `Git <https://git-scm.com/>`__;

   .. code:: bash

      $ git clone https://github.com/your_account_name/abess.git
      $ cd abess

4. Create a new branch, e.g. named ``my_branch``, to hold your
   development changes:

   .. code:: bash

      $ git branch my_branch
      $ git checkout my_branch

   It is preferred to work on your own branch instead of the master one;

5. While developing code, make sure to read the abess style guide (PEP8
   for Python, tidyverse for R) which will make sure that your new code
   and documentation matches the existing style. This makes the review
   process much smoother. For more details about code developing, read
   the `Code Developing <CodeDeveloping.md>`__ description for abess
   library;

6. After finishing the development and making sure it works well, you
   can push them onto your repository:

   .. code:: bash

      $ git add changed_files
      $ git commit -m "some commits"
      $ git push

7. Look back to GItHub, merge your branch with the master branch that
   have the up-to-date codes in
   `abess <https://github.com/abess-team/abess>`__; and click the
   “Contribute” button on your fork to open pull request. Now, we can
   receive your contribution.
