# Linear Algebra

By now, we can load datasets into tensors and manipulate these tensors with
basic mathematical operations.  To start building sophisticated models, we will
also need a few tools from linear algebra.  This section offers a gentle
introduction to the most essential concepts, starting from scalar arithmetic and
ramping up to matrix multiplication.

## Scalars

Most everyday mathematics consists of manipulating numbers one at a time.
Formally, we call these values _scalars_.  For example, the temperature in
Orlando, Florida is a balmy $$72$$ degrees Fahrenheit.  If you wanted to convert
the temperature to Celsius you would evaluate the expression $$c = \frac{5}{9}(f - 32)$$,
setting $$f$$ to $$72$$.  In this equation, the values $$5$$, $$9$$,
and $$32$$ are constant scalars.  The variables $$c$$ and $$f$$ in general
represent unknown scalars.

We denote scalars by ordinary lower-cased letters (e.g., $$x$$, $$y$$, and
$$z$$) and the space of all (continuous) _real-valued_ scalars by
$$\mathbb{R}$$.  For expedience, we will skip past rigorous definitions of
_spaces_: just remember that the expression $$x \in \mathbb{R}$$ is a formal way
to say that $$x$$ is a real-valued scalar.  The symbol $$\in$$ (pronounced "in")
denotes membership in a set.  For example, $$x, y \in \{0, 1\}$$ indicates that
$$x$$ and $$y$$ are variables that can only take values $$0$$ or $$1$$.

Scalars are implemented as tensors that contain only one element. Below, we
assign two scalars and perform the familiar addition, multiplication, division,
and exponentiation operations.

```perl
pdl> $x = ones(1) * 3.0
pdl> $y = ones(1) * 2.0
pdl> print $x, $y
[3] [2]
pdl> print $x+$y
pdl> print $x + $y, $x * $y, $x / $y, $x ** $y
[5] [6] [1.5] [9]
```

## Vectors

For current purposes, _you can think of a vector as a fixed-length array of
scalars_.  As with their code counterparts, we call these scalars the _elements_
of the vector (synonyms include _entries_ and _components_).  When vectors
represent examples from real-world datasets, their values hold some real-world
significance.  For example, if we were training a model to predict the risk of a
loan defaulting, we might associate each applicant with a vector whose
components correspond to quantities like their income, length of employment, or
number of previous defaults.  If we were studying the risk of heart attack, each
vector might represent a patient and its components might correspond to their
most recent vital signs, cholesterol levels, minutes of exercise per day, etc.
We denote vectors by bold lowercase letters, (e.g., $$\mathbf{x}$$, $$\mathbf{y}$$,
and $$\mathbf{z}$$).

Vectors are implemented as $$1^{\textrm{st}}$$-order tensors.  In general, such
tensors can have arbitrary lengths, subject to memory limitations. **Caution**:
in Perl (and in `PDL`), as in most _reasonable_ programming languages, vector
indices start at $$0$$, also known as *zero-based indexing*, whereas in linear
algebra subscripts begin at $$1$$ (one-based indexing), in this document.

```perl
pdl> $x = sequence(3)
pdl> print $x
[0 1 2]
```

We can refer to an element of a vector by using a subscript.  For example,
$$x_2$$ denotes the second element of $$\mathbf{x}$$.  Since $$x_2$$ is a
scalar, we do not bold it.  By default, we visualize vectors by stacking their
elements vertically:

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$

Here $$x_1, \ldots, x_n$$ are elements of the vector.  Later on, we will
distinguish between such _column vectors_ and *row vectors* whose elements are
stacked horizontally.  Recall that _we access a tensor's elements via indexing_.

```perl
pdl> print $x(2)
[2]
```

To indicate that a vector contains $$n$$ elements, we write $$\mathbf{x} \in
\mathbb{R}^n$$.  Formally, we call $$n$$ the *dimensionality* of the vector.  In
code, this corresponds to the tensor's length, accessible via the `length`
function in `PDL`. The output of `length` is identical to `dims`, which is the
generic dimensionality retrieval function.  The return value of `dims` is a
tuple that indicates a tensor's length along each dimension.  _Tensors with just one
dimension have shapes with just one element._ We can also use the `shape` function
to return a `PDL` object with the dimensions.

```perl
pdl> print $x->length
3
pdl> print $x->dims
3
pdl> print $x->shape
[3]
```

Oftentimes, the word "dimension" gets overloaded to mean both the number of axes
and the length along a particular dimension.  To avoid this confusion, we use *order*
to refer to the number of axes and *dimensionality* exclusively to refer to the
number of components.

## Matrices

Just as scalars are $$0^{\textrm{th}}$$-order tensors and vectors are
$$1^{\textrm{st}}$$-order tensors, matrices are $$2^{\textrm{nd}}$$-order tensors.
We denote matrices by bold capital letters (e.g., $$\mathbf{X}$$, $$\mathbf{Y}$$,
and $$\mathbf{Z}$$), and represent them in code by tensors with two axes.  The
expression $$\mathbf{A} \in \mathbb{R}^{m \times n}$$ indicates that a matrix
$$\mathbf{A}$$ contains $$m \times n$$ real-valued scalars, arranged as $$m$$ rows and
$$n$$ columns.  When $$m = n$$, we say that a matrix is *square*.  Visually, we can
illustrate any matrix as a table.  To refer to an individual element, we
subscript both the row and column indices, e.g., $$a_{ij}$$ is the value that
belongs to $$\mathbf{A}$$'s $$i^{\textrm{th}}$$ row and $$j^{\textrm{th}}$$ column:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$


In code, we represent a matrix $$\mathbf{A} \in \mathbb{R}^{m \times n}$$ by a
$$2^{\textrm{nd}}$$-order tensor with shape ($$m$$, $$n$$).  _We can convert any
appropriately sized $$m \times n$$ tensor into an $$m \times n$$ matrix_ by
passing the desired shape to `reshape`. Recall that `PDL` requires the order to
be swapped since it uses column-major form:

```perl
pdl> $A = sequence(6)->reshape(2,3)
pdl> print $A
[
 [0 1]
 [2 3]
 [4 5]
]
```

Sometimes we want to flip the axes.  When we exchange a matrix's rows and
columns, the result is called its _transpose_.  Formally, we signify a matrix
$$\mathbf{A}$$'s transpose by $$\mathbf{A}^\top$$ and if $$\mathbf{B} =
\mathbf{A}^\top$$, then $$b_{ij} = a_{ji}$$ for all $$i$$ and $$j$$.  Thus, the
transpose of an $$m \times n$$ matrix is an $$n \times m$$ matrix:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

In code, we can access _any matrix's transpose_ using the `transpose` function
as shown below:

```perl
pdl> print $A->transpose
[
 [0 2 4]
 [1 3 5]
]
```

Symmetric matrices are the subset of square matrices that are equal to their own
transposes:
$$\mathbf{A} = \mathbf{A}^\top$$.
The following matrix is symmetric:

```perl
pdl> $A = pdl([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
pdl> print $A == $A->transpose

[
 [1 1 1]
 [1 1 1]
 [1 1 1]
]

```

Matrices are useful for representing datasets.  Typically, rows correspond to
individual records and columns correspond to distinct attributes.

## Tensors

While you can go far in your machine learning journey with only scalars,
vectors, and matrices, eventually you may need to work with higher-order
_tensors_.  Tensors give us a generic way of describing extensions to
$$n^{\textrm{th}}$$-order arrays. We call software objects of the _tensor
class_ "tensors" precisely because they too can have arbitrary numbers of axes.
While it may be confusing to use the word _tensor_ for both the mathematical
object and its realization in code, our meaning should usually be clear from
context.  We denote general tensors by capital letters with a special font face
(e.g., $$\mathsf{X}$$, $$\mathsf{Y}$$, and $$\mathsf{Z}$$) and their indexing
mechanism (e.g., $$x_{ijk}$$ and $$[\mathsf{X}]_{1, 2i-1, 3}$$) follows
naturally from that of matrices.

Tensors will become more important when we start working with images.  Each
image arrives as a $$3^{\textrm{rd}}$$-order tensor with axes corresponding to
the height, width, and _channel_.  At each spatial location, the intensities of
each color (red, green, and blue) are stacked along the channel.  Furthermore, a
collection of images is represented in code by a $$4^{\textrm{th}}$$-order
tensor, where distinct images are indexed along the first dimension.  Higher-order
tensors are constructed, as were vectors and matrices, by growing the number of
shape components.

**NOTE**: In `PDL` the order of the parameters to `reshape` are reverse that of in
the `Python` equivalent.

```perl
pdl> print sequence(24)->reshape(4,3,2)
[
 [
  [ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]
 ]
 [
  [12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]
 ]
]

```

## Basic Properties of Tensor Arithmetic

Scalars, vectors, matrices, and higher-order tensors all have some handy
properties.  For example, elementwise operations produce outputs that have the
same shape as their operands.

```perl
pdl> $A = sequence(6)->reshape(3,2)
### Assign a copy of $A to $B by allocating new memory
pdl> $B = $A->copy
pdl> print $A, $A+$B

[
 [0 1 2]
 [3 4 5]
]

[
 [ 0  2  4]
 [ 6  8 10]
]
```

The elementwise product of two matrices is called their *Hadamard product*
(denoted $$\odot$$).  We can spell out the entries of the Hadamard product of
two matrices $$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$$:



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```perl
pdl> print $A * $B

[
 [ 0  1  4]
 [ 9 16 25]
]
```

Adding or multiplying a scalar and a tensor produces a result
with the same shape as the original tensor.
Here, each element of the tensor is added to (or multiplied by) the scalar.

```perl
pdl> print $a + $X, ($a * $X)->shape

[
 [
  [ 2  3  4  5]
  [ 6  7  8  9]
  [10 11 12 13]
 ]
 [
  [14 15 16 17]
  [18 19 20 21]
  [22 23 24 25]
 ]
]
 [4 3 2]

```

## Reduction

Often, we wish to calculate the sum of a tensor's elements.  To express the sum
of the elements in a vector $$\mathbf{x}$$ of length $$n$$, we write
$$\sum_{i=1}^n x_i$$. There is a simple function for it:

```perl
pdl> $x = sequence(3)
pdl> print $x, $x->sum
[0 1 2] 3
```

To express sums over the elements of tensors of arbitrary shape, we simply sum
over all its axes.  For example, the sum of the elements of an $$m \times n$$
matrix $$\mathbf{A}$$ could be written $$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$$.

```perl
pdl> print $A

[
 [0 1 2]
 [3 4 5]
]

pdl> print $A->shape, $A->sum
[3 2] 15
```

By default, invoking the sum function _reduces_ a tensor along all of its axes,
eventually producing a scalar.  The `sumover` function allows us to specify the
axes along which the tensor should be reduced.  To sum over all elements along
the rows (dimension 0), we call `sumover` with no arguments.  Since the input
matrix reduces along dimension 0 to generate the output vector, this dimension
is missing from the shape of the output.

```perl
pdl> print $A->sumover, $A->sumover->shape
[3 12] [2]
```

If we want to reduce along the columns we call the `mv` function with the
dimension we want to swap and call `sumover` on that. Here we _move_ dimensions
0 and 1 and invoke `sumover` on the result.

```perl
pdl> print $A->mv(0,1)->sumover
[3 5 7]
## what the move looks like
pdl> print $A->mv(0,1)

[
 [0 3]
 [1 4]
 [2 5]
]
```

Another way is to use `transpose` with `sumover`.

```perl
pdl> print $A->transpose->sumover
[3 5 7]
```

Reducing a matrix along both rows and columns via summation is equivalent to
summing up all the elements of the matrix.

```perl
pdl> print $A->sumover
[3 12]
pdl> print $A->sumover->sum
15
```

A related quantity is the _mean_, also called the _average_.  We calculate the
mean by dividing the sum by the total number of elements.  Because computing the
mean is so common, it gets a dedicated library function that works analogously
to `sum`. In `PDL` this function is `avg`. Likewise, the function `average` can
also calculate the mean along specific dimensions.

```perl
pdl> print $A->avg
2.5
pdl> print $A->average
[1 4]
```

## Non-Reduction Sum

Sometimes it can be useful to keep the number of dimensions unchanged when
invoking the function for calculating the sum or mean. This matters when we
want to use the broadcast mechanism.

```perl
pdl> print $A
[
 [1 2 3]
 [2 0 4]
 [3 4 5]
]
pdl> print $A->sumover
[6 6 12]
pdl> $sumA = $A->sumover->transpose
pdl> print $sumA
[
 [ 6]
 [ 6]
 [12]
]
```

For instance, since `$sumA` keeps its two axes after summing each row, we can
divide `$A` by `$sumA` with broadcasting to create a matrix where each row sums
up to $$1$$.

```perl
pdl> print $A/$sumA

[
 [       0.16666667        0.33333333               0.5]
 [       0.33333333                 0        0.66666667]
 [             0.25        0.33333333        0.41666667]
]

```

If we want to calculate the cumulative sum of elements of `$A` along some dimension,
say `dimension 0` (row by row), we can call the `cumusumover` function.
By design, this function does not reduce the input tensor along any dimension.

```perl
pdl> print $A->cumusumover
[
 [ 1  3  6]
 [ 2  2  6]
 [ 3  7 12]
]
```

## Dot Products

So far, we have only performed elementwise operations, sums, and averages.  And
if this was all we could do, linear algebra would not deserve its own section.
Fortunately, this is where things get more interesting.  One of the most
fundamental operations is the dot product.  Given two vectors $$\mathbf{x},
\mathbf{y} \in \mathbb{R}^d$$, their *dot product* $$\mathbf{x}^\top
\mathbf{y}$$ (also known as *inner product*, $$\langle \mathbf{x}, \mathbf{y}
\rangle$$) is a sum over the products of the elements at the same position:
$$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$$.

The *dot product* of two vectors is a sum over the products of the elements at
the same position.

In `PDL`, the _dot product_ can be obtained by using the `inner` function for
two vectors, since it is also known as the _inner product_ of two vectors.

```perl
pdl> $x = sequence(3)
pdl> $y = ones(3)
pdl> print inner($x,$y)
3
```

Equivalently, we can calculate the dot product of two vectors
by performing an elementwise multiplication followed by a sum:

```perl
pdl> print sum($x * $y)
3
```

Dot products are useful in a wide range of contexts.  For example, given some
set of values, denoted by a vector $$\mathbf{x}  \in \mathbb{R}^n$$, and a set
of weights, denoted by $$\mathbf{w} \in \mathbb{R}^n$$, the weighted sum of the
values in $$\mathbf{x}$$ according to the weights $$\mathbf{w}$$ could be
expressed as the dot product $$\mathbf{x}^\top \mathbf{w}$$.  When the weights
are nonnegative and sum to $$1$$, i.e., $$\left(\sum_{i=1}^{n} {w_i} =
1\right)$$, the dot product expresses a *weighted average*.  After normalizing
two vectors to have unit length, the dot products express the cosine of the
angle between them.  Later in this section, we will formally introduce this
notion of *length*.


## Matrix--Vector Products

Now that we know how to calculate dot products, we can begin to understand the
*product* between an $$m \times n$$ matrix $$\mathbf{A}$$ and an $$n$$-dimensional
vector $$\mathbf{x}$$.  To start off, we visualize our matrix in terms of its row
vectors

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

where each $$\mathbf{a}^\top_{i} \in \mathbb{R}^n$$ is a row vector representing
the $$i^\textrm{th}$$ row of the matrix $$\mathbf{A}$$.

The matrix--vector product $$\mathbf{A}\mathbf{x}$$ is simply a column vector of
length $$m$$, whose $$i^\textrm{th}$$ element is the dot product
$$\mathbf{a}^\top_i \mathbf{x}$$:

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

We can think of multiplication with a matrix $$\mathbf{A}\in \mathbb{R}^{m
\times n}$$ as a transformation that projects vectors from $$\mathbb{R}^{n}$$ to
$$\mathbb{R}^{m}$$.  These transformations are remarkably useful.  For example,
we can represent rotations as multiplications by certain square matrices.
Matrix--vector products also describe the key calculation involved in computing
the outputs of each layer in a neural network given the outputs from the
previous layer.

To express a matrix--vector product in code, we use the same `inner` function.
The operation is inferred based on the type of the arguments.  Note that the
column dimension of `$A` (its length along dimension 1) must be the same as the
dimension of `$x` (its length).

```perl
pdl> print $x
[0 1 2]
pdl> print $A
[
 [ 0  2  6]
 [ 0  0  8]
 [ 0  4 10]
]
pdl> print $A->shape, $x->shape
[3 3] [3]
pdl> print inner($A, $x)
[8 8 14]
```

## Matrix--Matrix Multiplication

Once you have gotten the hang of dot products and matrix--vector products, then
*matrix--matrix multiplication* should be straightforward.

Say that we have two matrices $$\mathbf{A} \in \mathbb{R}^{n \times k}$$ and
$$\mathbf{B} \in \mathbb{R}^{k \times m}$$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Let $$\mathbf{a}^\top_{i} \in \mathbb{R}^k$$ denote the row vector representing
the $$i^\textrm{th}$$ row of the matrix $$\mathbf{A}$$ and let $$\mathbf{b}_{j} \in
\mathbb{R}^k$$ denote the column vector from the $$j^\textrm{th}$$ column of the
matrix $$\mathbf{B}$$:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


To form the matrix product $$\mathbf{C} \in \mathbb{R}^{n \times m}$$, we simply
compute each element $$c_{ij}$$ as the dot product between the $$i^{\textrm{th}}$$
row of $$\mathbf{A}$$ and the $$j^{\textrm{th}}$$ column of $$\mathbf{B}$$, i.e.,
$$\mathbf{a}^\top_i \mathbf{b}_j$$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

We can think of the matrix--matrix multiplication $$\mathbf{AB}$$ as performing
$$m$$ matrix--vector products or $$m \times n$$ dot products and stitching the
results together to form an $$n \times m$$ matrix.  In the following snippet, we
perform matrix multiplication on `$A` and `$B`.  Here, `$A` is a matrix with two
rows and three columns, and `$B` is a matrix with three rows and four columns.
After multiplication, we obtain a matrix with two rows and four columns.  In
`PDL`, this can be accomplished by the operator `x` or the function `matmult`.

```
pdl> $A = sequence(6)->reshape(3,2)
[
 [0 1 2]
 [3 4 5]
]
pdl> $B = ones(4,3)
pdl> print $B

[
 [1 1 1 1]
 [1 1 1 1]
 [1 1 1 1]
]
pdl> print matmult($A,$B)
[
 [ 3  3  3  3]
 [12 12 12 12]
]
pdl> print $A x $B
[
 [ 3  3  3  3]
 [12 12 12 12]
]

```

The term *matrix--matrix multiplication* is often simplified to *matrix
multiplication*, and should not be confused with the Hadamard product.


## Norms

Some of the most useful operators in linear algebra are *norms*.  Informally,
the norm of a vector tells us how *big* it is.  For instance, the $$\ell_2$$
norm measures the (Euclidean) length of a vector.  Here, we are employing a
notion of *size* that concerns the magnitude of a vector's components (not its
dimensionality).

A norm is a function $$\| \cdot \|$$ that maps a vector to a scalar and
satisfies the following three properties:

1. Given any vector $$\mathbf{x}$$, if we scale (all elements of) the vector
   by a scalar $$\alpha \in \mathbb{R}$$, its norm scales accordingly:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. For any vectors $$\mathbf{x}$$ and $$\mathbf{y}$$:
   norms satisfy the triangle inequality:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. The norm of a vector is nonnegative and it only vanishes if the vector is zero:
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

Many functions are valid norms and different norms encode different notions of
size.  The Euclidean norm that we all learned in elementary school geometry when
calculating the hypotenuse of a right triangle is the square root of the sum of
squares of a vector's elements.  Formally, this is called the $$\ell_2$$ _norm_
and expressed as

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$

The method `magnover` calculates the $$\ell_2$$ norm in `PDL`. This is different
from the `norm` function in `PDL` which
[normalizes](https://metacpan.org/pod/PDL::Primitive#norm) a vector.
For matrices, in `PDL` you have to use the `mnorm` function available in
the `PDL::LinearAlgebra` module. So maybe it makes sense to use `mnorm` for
vectors too.

```perl
pdl> print pdl([3, -4])->magnover
5
pdl> use PDL::LinearAlgebra
pdl> print pdl([3, -4])->mnorm
5
```

The $$\ell_1$$ norm is also common and the associated measure is called the
Manhattan distance.  By definition, the $$\ell_1$$ norm sums the absolute values
of a vector's elements:

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

Compared to the $$\ell_2$$ norm, it is less sensitive to outliers.  To compute
the $$\ell_1$$ norm, we compose the absolute value with the sum operation.

```perl
pdl> print pdl([[3, -4]])->abs->sumover
[7]
```

Both the $$\ell_2$$ and $$\ell_1$$ norms are special cases of the more general
$$\ell_p$$ _norms_:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

In the case of matrices, matters are more complicated.  After all, matrices can
be viewed both as collections of individual entries _and_ as objects that
operate on vectors and transform them into other vectors.  For instance, we can
ask by how much longer the matrix--vector product $$\mathbf{X} \mathbf{v}$$
could be relative to $$\mathbf{v}$$.  This line of thought leads to what is
called the *spectral* norm.  For now, we introduce the *Frobenius norm*, which
is much easier to compute and defined as the square root of the sum of the
squares of a matrix's elements:

$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

The Frobenius norm behaves as if it were an $$\ell_2$$ norm of a matrix-shaped
vector.  Invoking the following function will calculate the Frobenius norm of a
matrix.

```perl
pdl> use PDL::LinearAlgebra
pdl> print ones(9,4)->mnorm
6
```

While we do not want to get too far ahead of ourselves, we already can plant
some intuition about why these concepts are useful.  In deep learning, we are
often trying to solve optimization problems: _maximize_ the probability assigned
to observed data; _maximize_ the revenue associated with a recommender model;
_minimize_ the distance between predictions and the ground truth observations;
_minimize_ the distance between representations of photos of the same person
while _maximizing_ the distance between representations of photos of different
people.  These distances, which constitute the objectives of deep learning
algorithms, are often expressed as norms.


## Discussion

In this section, we have reviewed all the linear algebra that you will need to
understand a significant chunk of modern deep learning.  There is a lot more to
linear algebra, though, and much of it is useful for machine learning.  For
example, matrices can be decomposed into factors, and these decompositions can
reveal low-dimensional structure in real-world datasets.  There are entire
subfields of machine learning that focus on using matrix decompositions and
their generalizations to high-order tensors to discover structure in datasets
and solve prediction problems.  But this book focuses on deep learning.  And we
believe you will be more inclined to learn more mathematics once you have gotten
your hands dirty applying machine learning to real datasets.  So while we
reserve the right to introduce more mathematics later on, we wrap up this
section here.

If you are eager to learn more linear algebra, there are many excellent books
and online resources.  For a more advanced crash course, consider checking out
[Introduction to Linear Algebra by Strang
(1993)](https://www.abebooks.com/Introduction-Linear-Algebra-Gilbert-Strang-Brand/31821496156/bd),
[Linear Algebra Review and Reference by Kolter
(2008)](http://cs229.stanford.edu/section/cs229-linalg.pdf) and [The Matrix
Cookbook by Petersen & Pedersen
(2008)](https://www.cs.toronto.edu/~bonner/courses/2018s/csc338/matrix_cookbook.pdf)([archive.org
link to
pdf](https://archive.org/download/K_B_Petersen_and_M_S_Peders__The_Matrix_Cookbook/matrixcookbook.pdf)).

To recap:

* Scalars, vectors, matrices, and tensors are
  the basic mathematical objects used in linear algebra
  and have zero, one, two, and an arbitrary number of axes, respectively.
* Tensors can be sliced or reduced along specified axes
  via indexing, or operations such as `sum` and `mean`, respectively.
* Elementwise products are called Hadamard products.
  By contrast, dot products, matrix--vector products, and matrix--matrix products
  are not elementwise operations and in general return objects
  having shapes that are different from the the operands.
* Compared to Hadamard products, matrix--matrix products
  take considerably longer to compute (cubic rather than quadratic time).
* Norms capture various notions of the magnitude of a vector (or matrix),
  and are commonly applied to the difference of two vectors
  to measure their distance apart.
* Common vector norms include the $$\ell_1$$ and $$\ell_2$$ norms,
   and common matrix norms include the *spectral* and *Frobenius* norms.


## Exercises

1. Prove that the transpose of the transpose of a matrix is the matrix itself: $$(\mathbf{A}^\top)^\top = \mathbf{A}$$.
1. Given two matrices $$\mathbf{A}$$ and $$\mathbf{B}$$, show that sum and transposition commute: $$\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$$.
1. Given any square matrix $$\mathbf{A}$$, is $$\mathbf{A} + \mathbf{A}^\top$$ always symmetric? Can you prove the result by using only the results of the previous two exercises?
1. We defined the tensor `X` of shape (4, 3, 2) in this section. What is the output of `length(X)`? Write your answer without implementing any code, then check your answer using code.
1. For a tensor `X` of arbitrary shape, does `length(X)` always correspond to the length of a certain dimension of `X`? What is that dimension?
1. Run `$A / $A->sum()` and see what happens. Can you analyze the results?
1. When traveling between two points in downtown Manhattan, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you travel diagonally?
1. Consider a tensor of shape (4, 3, 2). What are the shapes of the summation outputs along dimensions 0, 1, and 2?
1. Feed a tensor with three or more dimensions to the `mnorm` function and observe its output. What does this function compute for tensors of arbitrary shape?
1. Consider three large matrices, say $$\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$$, $$\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$$ and $$\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$$, initialized with Gaussian random variables. You want to compute the product $$\mathbf{A} \mathbf{B} \mathbf{C}$$. Is there any difference in memory footprint and speed, depending on whether you compute $$(\mathbf{A} \mathbf{B}) \mathbf{C}$$ or $$\mathbf{A} (\mathbf{B} \mathbf{C})$$? Why?
1. Consider three large matrices, say $$\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$$, $$\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$$ and $$\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$$. Is there any difference in speed depending on whether you compute $$\mathbf{A} \mathbf{B}$$ or $$\mathbf{A} \mathbf{C}^\top$$? Why? What changes if you initialize $$\mathbf{C} = \mathbf{B}^\top$$ without cloning memory? Why?
1. Consider three matrices, say $$\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$$. Construct a tensor with three axes by stacking $$[\mathbf{A}, \mathbf{B}, \mathbf{C}]$$. What is the dimensionality? Slice out the second coordinate of the third dimension to recover $$\mathbf{B}$$. Check that your answer is correct.

[Next - Calculus](calculus.md)
