# Code Format

## CodeFactor

We check the C++, R and Python format by [CodeFactor](https://www.codefactor.io/repository/github/abess-team/abess). 
More specifically, the formatters and rules are:

| Language | Formatter                                                             | Rules                                                                              |
| -------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| C++      | [CppLint](https://github.com/google/styleguide/tree/gh-pages/cpplint) | [`abess/CPPLINT.cfg`](https://github.com/abess-team/abess/blob/master/CPPLINT.cfg) |
| Python   | [Pylint](https://www.pylint.org/)                                     | [`abess/.pylintrc`](https://github.com/abess-team/abess/blob/master/.pylintrc)     |
| R        | [Lintr](https://github.com/r-lib/lintr)                               | [`abess/.lintr`](https://github.com/abess-team/abess/blob/master/.lintr)           |

Each pull request will be checked, and some recommendations will be given if not passed. But don't be worry about those complex rules, most of them can be formatted automatically by some tools.

> Note that there may be few problems that the auto-fix tools can NOT deal with. In that situation, please update your pull request following the suggestions by CodeFactor.

## Auto-format

### C++

[Clang-Format](https://clang.llvm.org/docs/ClangFormat.html) is a powerful tool to format C/C++ code. You can install it quickly:

- Linux: `$ sudo apt install clang-format`;
- MacOS: `$ brew install clang-format`;
- Windows: download it from [LLVM](https://llvm.org/builds/);

#### with VS Code

If you use [Visual Studio Code](https://code.visualstudio.com/) for coding, an extension called [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) supports auto-fix by Clang-Format. 

However, in order to adapt to our rules, you need to add some statements in `setting.json` (the configuration file for Visual Studio Code):

```javascript
"C_Cpp.clang_format_fallbackStyle": "{BasedOnStyle: Google, UseTab: Never, IndentWidth: 4, TabWidth: 4, ColumnLimit: 120, Standard: Cpp11}",
"files.insertFinalNewline": true,
"files.trimFinalNewlines": true,
// "editor.formatOnSave": true  // enable auto-fix after saving a file
```

After that, you can right-click on an C++ file and then click "Format Document". That will be done.

#### with command line

Besides, Clang-Format supports using directly in command line or based on a configuration file. You can check them [here](https://clang.llvm.org/docs/ClangFormatStyleOptions.html). The configuration is similar as above:

```yaml
# `.clang-format` in the same directory of your C++ files
BasedOnStyle: Google
UseTab: Never
IndentWidth: 4
TabWidth: 4
ColumnLimit: 120
Standard: Cpp11
```

And then run `$ clang-format -style=file some_code.cpp > some_code_formatted.cpp` in command line. The formatted code is stored in `some_code_formatted.cpp` now.

### Python

[Autopep8](https://pypi.org/project/autopep8/) can be used in formatting Python code. You can easily install it by `$ pip install autopep8`.

#### with VS Code

Visual Studio Code can deal with Python auto-fix too, with [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) extension.

There is no more steps to do. Right-click on an Python file and then click "Format Document". That will be done.


#### with command line

As we memtioned above, the default setting of Autopep8 is enough for us. Hence run `$ autopep8 some_code.py > some_code_formatted.py` and the formatted code is stored in `some_code_formatted.py` now.
