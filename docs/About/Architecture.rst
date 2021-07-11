# Architecture

In this page, we briefly introduce our frame for fast and polynomial best subset selection:

![](./fig/frame.svg)

The core code of abess is built with C++ and the figure above shows the software architecture of abess and each building block will be described as follows. 

- **The Data class** accept tabular data and return a Data object used on other parts; 
- **The Algorithm class**, as the core class in abess, implement the generic splicing procedure for best subset selection. Seven built-in tasks are present and you can also add your algorithm as the next section shows. 
- **The Metric class** serves as a evaluator. Based on the Algorithm and Data objects, it evaluate the estimation at a given support size by cross validation or information criterion. 
- Finally, **R or Python interface** collects the results.

For more details, please read *[这里放文献链接？（如果有）]()*

