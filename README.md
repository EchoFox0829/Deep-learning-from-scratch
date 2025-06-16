Deep Learning from Scratch 2
============================

[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-2/images/deep-learning-from-scratch-2.png" width="200px">](https://www.oreilly.co.jp/books/9784873118369/)

This is the support site for the book “[Deep Learning from Scratch 2: Natural Language Processing Edition](https://www.oreilly.co.jp/books/9784873118369/)” (O'Reilly Japan). The source code used in the book is collected here.


## Directory Structure

| Folder    | Description                        |
|:--        |:--                                 |
| ch01      | Source code for Chapter 1          |
| ch02      | Source code for Chapter 2          |
| ...       | ...                                |
| ch08      | Source code for Chapter 8          |
| common    | Common source code                 |
| dataset   | Source code for datasets           |

Pre-trained weight files (used in Chapters 6 and 7) can be downloaded from the following URL:  
<https://www.oreilly.co.jp/pub/9784873118369/BetterRnnlm.pkl>

For explanations of the source code, please refer to the book.


## Python and External Libraries

To run the source code, you need the following software:

* Python 3.x (version 3 series)
* NumPy
* Matplotlib

Optionally, you can also use the following libraries:

* SciPy (optional)
* CuPy (optional)

## How to Run

Move to the folder of each chapter and run the Python command.

```
$ cd ch01
$ python train.py

$ cd ../ch05
$ python train_custom_loop.py
```

## License

The source code in this repository is licensed under the [MIT License](http://www.opensource.org/licenses/MIT).  
You are free to use it for both commercial and non-commercial purposes.

## Errata

Errata for the book are published at the following page:

https://github.com/oreilly-japan/deep-learning-from-scratch-2/wiki/errata

If you find any errors not listed on this page, please contact [japan@oreilly.co.jp](<mailto:japan@oreilly.co.jp>).