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
by supplying (possibly nested) Python list(s) 
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

As with  Perl/Python lists,
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
two matrices along rows (axis 0 in Python, dim 1 in `PDL`)
instead of columns (axis 1 in Python, dim 0 in `PDL`).
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
:label:`subsec_broadcasting`

By now, you know how to perform 
elementwise binary operations
on two tensors of the same shape. 
Under certain conditions,
even when shapes differ, 
we can still [**perform elementwise binary operations
by invoking the *broadcasting mechanism*.**]
Broadcasting works according to 
the following two-step procedure:
(i) expand one or both arrays
by copying elements along axes with length 1
so that after this transformation,
the two tensors have the same shape;
(ii) perform an elementwise operation
on the resulting arrays.

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

Since `a` and `b` are $$3\times1$$ 
and $$1\times2$$ matrices, respectively,
their shapes do not match up.
Broadcasting produces a larger $$3\times2$$ matrix 
by replicating matrix `a` along the columns
and matrix `b` along the rows
before adding them elementwise.

```{.python .input}
%%tab all
a + b
```

## Saving Memory

[**Running operations can cause new memory to be
allocated to host results.**]
For example, if we write `Y = X + Y`,
we dereference the tensor that `Y` used to point to
and instead point `Y` at the newly allocated memory.
We can demonstrate this issue with Python's `id()` function,
which gives us the exact address 
of the referenced object in memory.
Note that after we run `Y = Y + X`,
`id(Y)` points to a different location.
That is because Python first evaluates `Y + X`,
allocating new memory for the result 
and then points `Y` to this new location in memory.

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
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

:begin_tab:`mxnet, pytorch`
Fortunately, (**performing in-place operations**) is easy.
We can assign the result of an operation
to a previously allocated array `Y`
by using slice notation: `Y[:] = <expression>`.
To illustrate this concept, 
we overwrite the values of tensor `Z`,
after initializing it, using `zeros_like`,
to have the same shape as `Y`.
:end_tab:

:begin_tab:`tensorflow`
`Variables` are mutable containers of state in TensorFlow. They provide
a way to store your model parameters.
We can assign the result of an operation
to a `Variable` with `assign`.
To illustrate this concept, 
we overwrite the values of `Variable` `Z`
after initializing it, using `zeros_like`,
to have the same shape as `Y`.
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# JAX arrays do not allow in-place operations
```

:begin_tab:`mxnet, pytorch`
[**If the value of `X` is not reused in subsequent computations,
we can also use `X[:] = X + Y` or `X += Y`
to reduce the memory overhead of the operation.**]
:end_tab:

:begin_tab:`tensorflow`
Even once you store state persistently in a `Variable`, 
you may want to reduce your memory usage further by avoiding excess
allocations for tensors that are not your model parameters.
Because TensorFlow `Tensors` are immutable 
and gradients do not flow through `Variable` assignments, 
TensorFlow does not provide an explicit way to run
an individual operation in-place.

However, TensorFlow provides the `tf.function` decorator 
to wrap computation inside of a TensorFlow graph 
that gets compiled and optimized before running.
This allows TensorFlow to prune unused values, 
and to reuse prior allocations that are no longer needed. 
This minimizes the memory overhead of TensorFlow computations.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be reused when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## Conversion to Other Python Objects

:begin_tab:`mxnet, tensorflow`
[**Converting to a NumPy tensor (`ndarray`)**], or vice versa, is easy.
The converted result does not share memory.
This minor inconvenience is actually quite important:
when you perform operations on the CPU or on GPUs,
you do not want to halt computation, waiting to see
whether the NumPy package of Python 
might want to be doing something else
with the same chunk of memory.
:end_tab:

:begin_tab:`pytorch`
[**Converting to a NumPy tensor (`ndarray`)**], or vice versa, is easy.
The torch tensor and NumPy array 
will share their underlying memory, 
and changing one through an in-place operation 
will also change the other.
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

To (**convert a size-1 tensor to a Python scalar**),
we can invoke the `item` function or Python's built-in functions.

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## Summary

The tensor class is the main interface for storing and manipulating data in deep learning libraries.
Tensors provide a variety of functionalities including construction routines; indexing and slicing; basic mathematics operations; broadcasting; memory-efficient assignment; and conversion to and from other Python objects.


## Exercises

1. Run the code in this section. Change the conditional statement `X == Y` to `X < Y` or `X > Y`, and then see what kind of tensor you can get.
1. Replace the two tensors that operate by element in the broadcasting mechanism with other shapes, e.g., 3-dimensional tensors. Is the result the same as expected?

