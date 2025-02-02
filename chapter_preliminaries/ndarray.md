# Data Manipulation

In order to get anything done, 
we need some way to store and manipulate data.
Generally, there are two important things 
we need to do with data: 
(i) acquire them; 
and (ii) process them once they are inside the computer. 
There is no point in acquiring data 
without some way to store it, 
so to start, let's get our hands dirty
with $$n$$-dimensional arrays, 
which we also call *tensors*.
If you already know the NumPy 
scientific computing package, 
this will be a breeze.
For all modern deep learning frameworks,
the *tensor class* (`ndarray` in MXNet, 
`Tensor` in PyTorch and TensorFlow) 
resembles NumPy's `ndarray`,
with a few killer features added.
First, the tensor class
supports automatic differentiation.
Second, it leverages GPUs
to accelerate numerical computation,
whereas NumPy only runs on CPUs.
These properties make neural networks
both easy to code and fast to run.


## Getting Started

To start, we can test this on the `perldl` interpreter that comes with
installing `PDL`, as that would make this process very easy.
This interpreter loads all the required functions you will need.

```bash
$ perldl
pdl> use PDL::AutoLoader
pdl>
```

**A tensor represents a (possibly multidimensional) array of numerical values.**
In the one-dimensional case, i.e., when only one axis is needed for the data,
a tensor is called a *vector*.
With two axes, a tensor is called a *matrix*.
With $$k > 2$$ axes, we drop the specialized names
and just refer to the object as a $$k^\textrm{th}$$-*order tensor*.

PDL provides a variety of functions 
for creating new tensors 
prepopulated with values. 
For example, by invoking `xvals(n)` or `sequence(n)`,
we can create a vector of evenly spaced values,
starting at 0 (included) 
and ending at `n` (not included).
By default, the interval size is $$1$$.
Unless otherwise specified, 
new tensors are stored in main memory 
and designated for CPU-based computation.

```perl
pdl> $x = xvals(12)
pdl> print $x 
[0 1 2 3 4 5 6 7 8 9 10 11]
pdl> $x = sequence 12
[0 1 2 3 4 5 6 7 8 9 10 11]
```

Each of these values is called
an *element* of the tensor.
The tensor `x` contains 12 elements.
We can inspect the total number of elements 
in a tensor via its `dims` attribute or using the `dims` function.

```perl
pdl> print $x->dims
12
pdl> print dims($x)
12
```

(**We can access a tensor's *shape***) 
(the length along each axis)
by inspecting its `shape` attribute.
Because we are dealing with a vector here,
the `shape` contains just a single element
and is identical to the size.

```perl
pdl> print $x->shape
[12]
```

We can [**change the shape of a tensor
without altering its size or values**],
by invoking `reshape`.
For example, we can transform 
our vector `x` whose shape is `[12]`
to a matrix `X`  with shape `(3, 4)`.
PDL stores the data in column major form, so we have to swap the rows and
columns.
This new tensor retains all elements
but reconfigures them into a matrix.
Notice that the elements of our vector
are laid out one row at a time and thus
`x[3] == X[3, 0]`.

```perl
pdl> $X = $x->reshape(4,3)
pdl> print $X
[
 [ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
]
pdl> print $X(3,0)
[
 [3]
]
pdl> print $X(3)
[
 [ 3]
 [ 7]
 [11]
]
```

Note that specifying every shape component
to `reshape` is redundant.
Because we already know our tensor's size,
we can work out one component of the shape given the rest.
For example, given a tensor of size $$n$$
and target shape ($$h$$, $$w$$),
we know that $$w = n/h$$.

Practitioners often need to work with tensors
initialized to contain all 0s or 1s.
[**We can construct a tensor with all elements set to 0**] (~~or one~~)
and a shape of (4, 3, 2) via the `zeroes` function.

```perl
pdl> print zeroes(4,3,2)
[
 [
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
 ]
 [
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
 ]
]
```

Similarly, we can create a tensor 
with all 1s by invoking `ones`.

```perl
pdl> print ones(4,3,2)

[
 [
  [1 1 1 1]
  [1 1 1 1]
  [1 1 1 1]
 ]
 [
  [1 1 1 1]
  [1 1 1 1]
  [1 1 1 1]
 ]
]
```

We often wish to 
[**sample each element randomly (and independently)**] 
from a given probability distribution.
For example, the parameters of neural networks
are often initialized randomly.
The following snippet creates a tensor 
with elements drawn from 
a standard Gaussian (normal) distribution
with mean 0 and standard deviation 1.

```perl
pdl> print grandom(4,3)

[
 [        -0.63968335          0.42479337         -0.81623105        -0.010018838]
 [        -0.34909049          0.57365255          0.32526079          0.68310597]
 [          1.0762051           2.3493898          0.53131591          -1.1742487]
]
```

Finally, we can construct tensors by
[**supplying the exact values for each element**] 
by supplying (possibly nested) Perl list(s) 
containing numerical literals.
Here, we construct a matrix with a list of lists,
where the outermost list corresponds to axis 0,
and the inner list corresponds to axis 1.

```perl
pdl> print pdl([[2,1,4,3],[1,2,3,4],[4,3,2,1]])

[
 [2 1 4 3]
 [1 2 3 4]
 [4 3 2 1]
]
```

## Indexing and Slicing

As with  Perl lists,
we can access tensor elements 
by indexing (starting with 0).
To access an element based on its position
relative to the end of the list,
we can use negative indexing using `PDL::NiceSlice`.
Finally, we can access whole ranges of indices 
via slicing (e.g., `X[start:stop]`), 
where the returned value includes 
the first index (`start`) *but not the last* (`stop`).
Finally, when only one index (or slice)
is specified for a $$k^\textrm{th}$$-order tensor,
it is applied along axis 0.
Thus, in the following code,
[**`(,-1)` selects the last row and `(,1:2)`
selects the second and third rows**].
You can select columns like `(1,-1)` and `(1,1:2)`

```perl
pdl> print $X(, 1:2)
[
 [ 4  5  6  7]
 [ 8  9 10 11]
]
pdl> print $X(, -1)
[
 [ 8  9 10 11]
]
## remove the extra nesting
pdl> print $X(:,(-1))
[8 9 10 11]
pdl> print $X(1,1:2)
[
 [5]
 [9]
]
pdl> print $X(1,-1)
[
 [9]
]
pdl> print $X(1:1, (-1))
[9]
```

If we want [**to assign multiple elements the same value,
we apply the indexing on the left-hand side 
of the assignment operation.**]
For instance, `(:,0:1)`  accesses 
the first and second rows,
where `:` takes all the elements along axis 1 (column).
While we discussed indexing for matrices,
this also works for vectors
and for tensors of more than two dimensions.
The `.` operator does assignment in `PDL`.

```perl
pdl> $X(:,0:1) .= 12

pdl> print $X

[
 [12 12 12 12]
 [12 12 12 12]
 [ 8  9 10 11]
]
```

## Operations

Now that we know how to construct tensors
and how to read from and write to their elements,
we can begin to manipulate them
with various mathematical operations.
Among the most useful of these 
are the *elementwise* operations.
These apply a standard scalar operation
to each element of a tensor.
For functions that take two tensors as inputs,
elementwise operations apply some standard binary operator
on each pair of corresponding elements.
We can create an elementwise function 
from any function that maps 
from a scalar to a scalar.

In mathematical notation, we denote such
*unary* scalar operators (taking one input)
by the signature 
$$f: \mathbb{R} \rightarrow \mathbb{R}$$.
This just means that the function maps
from any real number onto some other real number.
Most standard operators, including unary ones like $$e^x$$, can be applied elementwise.

```perl
pdl> print $x->exp

[
 [               1        2.7182818        7.3890561        20.085537]
 [        54.59815        148.41316        403.42879        1096.6332]
 [        2980.958        8103.0839        22026.466        59874.142]
]

```

Likewise, we denote *binary* scalar operators,
which map pairs of real numbers
to a (single) real number
via the signature 
$$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$$.
Given any two vectors $$\mathbf{u}$$ 
and $$\mathbf{v}$$ *of the same shape*,
and a binary operator $$f$$, we can produce a vector
$$\mathbf{c} = F(\mathbf{u},\mathbf{v})$$
by setting $$c_i \gets f(u_i, v_i)$$ for all $$i$$,
where $$c_i, u_i$$, and $$v_i$$ are the $$i^\textrm{th}$$ elements
of vectors $$\mathbf{c}, \mathbf{u}$$, and $$\mathbf{v}$$.
Here, we produced the vector-valued
$$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$$
by *lifting* the scalar function
to an elementwise vector operation.
The common standard arithmetic operators
for addition (`+`), subtraction (`-`), 
multiplication (`*`), division (`/`), 
and exponentiation (`**`)
have all been *lifted* to elementwise operations
for identically-shaped tensors of arbitrary shape.

```perl
pdl> print sequence(4)
[0 1 2 3]
pdl> $x = 2 ** sequence(4)
pdl> print $x
[1 2 4 8]
pdl> $y = 2 * ones(4)
[2 2 2 2]
pdl> print $x + $y, $x - $y, $x * $y, $x / $y, $x ** $y
[3 4 6 10] [-1 0 2 6] [2 4 8 16] [0.5 1 2 4] [1 4 16 64]
```

In addition to elementwise computations,
we can also perform linear algebraic operations,
such as dot products and matrix multiplications.
We will elaborate on these
in the [linear algebra section](linear-algebra.md).

We can also [***concatenate* multiple tensors,**]
stacking them end-to-end to form a larger one.
We just need to provide a list of tensors
and tell the system along which axis to concatenate.
The example below shows what happens when we concatenate
two matrices along rows (axis 0 in numpy in Python, dim 1 in `PDL`)
instead of columns (axis 1 in numpy in Python, dim 0 in `PDL`).
In `PDL` the axes are swapped because of column-major notation.
We can see that the first output's axis-0 length ($$6$$)
is the sum of the two input tensors' axis-0 lengths ($$3 + 3$$);
while the second output's axis-1 length ($$8$$)
is the sum of the two input tensors' axis-1 lengths ($$4 + 4$$).

```perl
pdl> $x1 = sequence(12)->reshape(4,3)
pdl> print $x1

[
 [ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
]

pdl> $y1 = pdl([[2,1,4,3], [1,2,3,4],[4,3,2,1]])
pdl> print $y1

[
 [2 1 4 3]
 [1 2 3 4]
 [4 3 2 1]
]

## along the row axis-1 is dim-0
pdl> print $x1->glue(0, $y1)
[
 [ 0  1  2  3  2  1  4  3]
 [ 4  5  6  7  1  2  3  4]
 [ 8  9 10 11  4  3  2  1]
]

## along the row axis-0 is dim-1
pdl> print $x1->glue(1, $y1)
[
 [ 0  1  2  3  2  1  4  3]
 [ 4  5  6  7  1  2  3  4]
 [ 8  9 10 11  4  3  2  1]
]

```

Sometimes, we want to 
[**construct a binary tensor via *logical statements*.**]
Take `X == Y` as an example.
For each position `i, j`, if `X[i, j]` and `Y[i, j]` are equal, 
then the corresponding entry in the result takes value `1`,
otherwise it takes value `0`.

```perl
pdl> print $x1 == $y1

[
 [0 1 0 1]
 [0 0 0 0]
 [0 0 0 0]
]

```

[**Summing all the elements in the tensor**] yields a tensor with only one element.

```perl
pdl> print $x1->sum
66
```

## Broadcasting

By now, you know how to perform 
elementwise binary operations
on two tensors of the same shape. 
Under certain conditions,
even when shapes differ, 
we can still perform elementwise binary operations
by invoking the *broadcasting mechanism*.
Broadcasting works according to 
the following two-step procedure:
(i) expand one or both arrays
by copying elements along axes with length 1
so that after this transformation,
the two tensors have the same shape;
(ii) perform an elementwise operation
on the resulting arrays.

```perl
pdl> $a = sequence(3)->reshape(1,3)
pdl> print $a
[
 [0]
 [1]
 [2]
]
pdl> $b = sequence(2)->reshape(2,1)
pdl> print $b
[
 [0 1]
]
```

Since `a` and `b` are $$3\times1$$ 
and $$1\times2$$ matrices, respectively,
their shapes do not match up.
Broadcasting produces a larger $$3\times2$$ matrix 
by replicating matrix `a` along the columns
and matrix `b` along the rows
before adding them elementwise.

```perl
pdl> print $a0 + $b0

[
 [0 1]
 [1 2]
 [2 3]
]

```

## Saving Memory

[**Running operations can cause new memory to be
allocated to host results.**]
For example, if we write `Y = X + Y`,
we dereference the tensor that `Y` used to point to
and instead point `Y` at the newly allocated memory.
We can demonstrate this issue with PDL's `address()` function,
which gives us the exact address 
of the referenced object in memory.
Note that after we run `$Y = $Y + $X`,
`$Y->address` points to a different location.
That is because PDL first evaluates `$Y + $X`,
allocating new memory for the result 
and then points `$Y` to this new location in memory.

```perl
pdl> $before = $b0->address
pdl> print $before
94494728210000
pdl> $b0 = $b0 + $a0
pdl> print $b0->address
pdl> print $b0->address
94494728088496
```

This might be undesirable for two reasons.
First, we do not want to run around
allocating memory unnecessarily all the time.
In machine learning, we often have
hundreds of megabytes of parameters
and update all of them multiple times per second.
Whenever possible, we want to perform these updates *in place*.
Second, we might point at the 
same parameters from multiple variables.
If we do not update in place, 
we must be careful to update all of these references,
lest we spring a memory leak 
or inadvertently refer to stale parameters.

PDL has an `inplace` function that allows this but can be tricky to use.
To force in-place semantics you need to set the `inplace` flag using
`set_inplace()` call on the variable. Then you need to use the `.` operator to
assign the new values and maintain the same address.
We demonstrate this with the example below. This only works if the dimensions
are identical for the left-hand side and right-hand side PDL objects.

```perl
pdl> $x1 = sequence(12)->reshape(4,3)
pdl> $y1 = pdl([[2,1,4,3], [1,2,3,4],[4,3,2,1]])
pdl> print $x1

[
 [ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
]
pdl> print $y1

[
 [2 1 4 3]
 [1 2 3 4]
 [4 3 2 1]
]

pdl> print $y1->dims
4 3
pdl> $z1 = zeroes($y1->dims)
pdl> print $z1

[
 [0 0 0 0]
 [0 0 0 0]
 [0 0 0 0]
]
pdl> print $z1->address
94494728214992
pdl> $z1->set_inplace(1)
pdl> $z1->is_inplace()
1
pdl> $z1 .= $x1 + $y1
pdl> print $z1->address
94494728214992
```

If the value of `X` is not reused in subsequent computations,
we can also use `X .= X + Y` or `X += Y`
to reduce the memory overhead of the operation.

PDL uses automatic garbage collection and if a variable is not needed, you can
always set it to `undef` in Perl to automatically mark it for garbage
collection.

## Conversion to Other Perl Objects

Converting to a Perl object or vice versa, is easy.
The converted result does not share memory.
This minor inconvenience is actually quite important:
when you perform operations on the CPU or on GPUs,
you do not want to halt computation, waiting to see
whether the PDL package might want to be doing something else
with the same chunk of memory.

```perl
pdl> $x_arr = $x1->unpdl
pdl> print $x_arr
ARRAY(0x55f144b5d570)
pdl> print ref($x_arr)
ARRAY
pdl> $x_pdl = pdl($x_arr)
pdl> print $x_pdl

[
 [ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
]
pdl> print ref($x_pdl)
PDL
```

To convert data types we can use a qualifier on the creation.


```perl
pdl> $x_pdl = float ($x_arr)

pdl> print $x_pdl

[
 [ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
]

pdl> print $x_pdl->type
float
```

## Summary

The tensor class is the main interface for storing and manipulating data in deep learning libraries.
Tensors provide a variety of functionalities including construction routines; indexing and slicing; basic mathematics operations; broadcasting; memory-efficient assignment; and conversion to and from other Perl objects.


## Exercises

1. Run the code in this section. Change the conditional statement `X == Y` to `X < Y` or `X > Y`, and then see what kind of tensor you can get.

    _Solution:_

```perl
pdl> $x1 = sequence(12)->reshape(4,3)
pdl> print $x1

[
 [ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
]

pdl> $y1 = pdl([[2,1,4,3], [1,2,3,4],[4,3,2,1]])
pdl> print $y1

[
 [2 1 4 3]
 [1 2 3 4]
 [4 3 2 1]
]

pdl> print $x1 > $y1

[
 [0 0 0 0]
 [1 1 1 1]
 [1 1 1 1]
]

pdl> print $x1 < $y1

[
 [1 0 1 0]
 [0 0 0 0]
 [0 0 0 0]
]

```

2. Replace the two tensors that operate by element in the broadcasting mechanism with other shapes, e.g., 3-dimensional tensors. Is the result the same as expected?

    _Solution:_

```perl
pdl> $a = sequence(3)->reshape(1,3)
pdl> print $a
[
 [0]
 [1]
 [2]
]
pdl> $b = sequence(3)->reshape(2,1)
pdl> print $b
[
 [0 1 2]
]
pdl> print $a + $b

[
 [0 1 2]
 [1 2 3]
 [2 3 4]
]

```

[Next - Data Pre-processing](preprocessing.md) [Previous - Preliminaries](index.md)
