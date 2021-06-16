# Code Developing

> Before developing the code, please follow the [Installation](../Installation.md) and make sure the initial code works on your device.

## Contribution

Contributions are welcome, and please make an issue at [https://github.com/abess-team/abess/issues](https://github.com/abess-team/abess/issues). Every little bit helps, and credit will always be given.

### Bugs Report 

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Feedback

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.

## Quick start

Firstly, to start with our frame, a simple introduction is worthy to be shown.

![](./fig/frame.svg)

The core code of abess is built with C++ and the figure above shows the software architecture of abess and each building block will be described as follows. 

- **The Data class** accept tabular data and return a Data object used on other parts; 
- **The Algorithm class**, as the core class in abess, implement the generic splicing procedure for best subset selection. Seven built-in tasks are present and you can also add your algorithm as the next section shows. 
- **The Metric class** serves as a evaluator. Based on the Algorithm and Data objects, it evaluate the estimation at a given support size by cross validation or information criterion. 
- Finally, **R or Python interface** collects the results.

For more details, please read *[这里放文献链接？（如果有）]()*

## Core C++

The main files related to the core are in `abess/python/src`. Among them, some important files:

- `Algorithm.h` records the implement of each concrete algorithm; 
- `abess.cpp` contain the calling procedure.

If you want to add a new algorithm, both of them should be updated.



In `Algorithm.h`, The concrete algorithms are programmed in the subclass of Algorithm by rewriting the virtual function interfaces of class Algorithm. Besides, the implementation is modularized such that you can easily extend the package. 

>  The format of a new algorithm's name is "**abess+your_algorithm**", which means that using abess to solve the problem.

Take PCA as an example, the name should be `abessPCA`. Now we can define a concrete algorithm like: [[code link]]()

```Cpp
template <class T4>
class abessPCA : public Algorithm<...>
{
public:
    abessPCA(...) : Algorithm<...>::Algorithm(...){};
    ~abessPCA(){};
 
    void primary_model_fit(...){...};
        // solve the subproblem under given active set
        // record the sparse answer in variable "beta"

    double neg_loglik_loss(...){...};
        // define and compute loss under given active set
        // return the current loss

    void sacrifice(...){...};
        // define and compute sacrifice for all variables (both forward and backward)
        // record sacrifice in variable "bd"

    double effective_number_of_parameter(...){...};
		// return effective number of parameter
}
```
Note that `sacrifice` function here would compute "forward/backward sacrifices" and record them in `bd`.

- For active variable, the lower (backward) sacrifice is, the more likely it will be dropped;
- For inactive variable, the higher (forward) sacrifice is, the more likely it will come into use.

Since it can be quite different to compute sacrifices for different problem, you may need to derivate by yourself.



After that, turn to `abess.cpp` and you will find some `new` command like: [[code link]]()
```Cpp
if (model_type == 1)
{
    algorithm_uni_dense = new abessLm<...>(...);
}
else if ...
```
They are used to request memory and call the algorithms in `Algorithm.h`. Here you need to add your own algorithm and give it a unused `model_type` number, e.g. 7.

```Cpp
...
else if (model_type == 7) // indicates PCA
{
    algorithm_uni_dense = new abessPCA<...>(...);
}
```
Note that there is a small difference between single response variable and multiple response variables question:
```Cpp
...
else if (model_type == 123) // indicates a multiple response algorithm
{
    // name it "mul"
    algorithm_mul_dense = new abessXXX<...>(...);
}
```

What is more, the variable named `algorithm_list_uni_dense[i]` or `algorithm_list_mul_dense[i]` is similar to what we said above. They are for parallel computing. [[code link]]()

Hence, you should also add:

```Cpp
...
else if (model_type == 7)
{
    // for single response variable question, name it "uni"
    algorithm_list_uni_dense[i] = new abessPCA<...>(...);
}
```
or,
```Cpp
...
else if (model_type == 123)
{
    // for multiple response variables question, name it "mul"
    algorithm_list_mul_dense[i] = new abessXXX<...>(...);
}
```

Now your new method has been connected to the whole frame. In the next section, we focus on how to build R or Python package based on the core code.

## R & Python Package

### R Package



### Python Package

To make your code available for Python, `cd` into directory `abess/python` and run `python setup.py install`.
> If you receive an error said "*Can't create or remove files in install directory*", this may be caused by permission denied. The step below may help with it.
>
> - For Linux: run `sudo python setup.py install`.
> - For Windows: run the command as administrator.
> - For MacOS:

It may take a few minutes to install:

- if the installation throw some errors, it means that the C++ code may be wrong;
- if the installation runs without errors, it will finish with message like "*Finished processing dependencies for abess*". 

Now a file named `cabess.py` will be appeared in the directory `abess/python/src`, which help to link Python and C++. You need to move it into directory `abess/python/abess`.

Then open file `abess/python/abess/linear.py` and add a new class like those already exist: [[code link]]()
```Python
class abessPCA(bess_base): 
    """
    Here is some introduction.
    """
    def __init__(self, ...):
        super(abessXXX, self).__init__(
            algorithm_type="abess", 
            model_type="PCA", 
            ...
        )
```

Then, the final step is to link this Python class with the model type number (it has been defined in the [Core C++](#Core C++)). In `linear.py`, you can find somewhere like: 
```Python
if self.model_type == "Lm":
    model_type_int = 1
elif ...
```

Add:

```python
...
elif self.model_type == "PCA":
    model_type_int = 7   
```

After finished all changes before, run `python setup.py install` again and this time the installation would be finished quickly. 

Congratulation! Your work can now be used by:
```Python
from abess.linear import abessPCA
```