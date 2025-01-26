# Data Preprocessing

So far, we have been working with synthetic data that arrived in ready-made
tensors.  However, to apply deep learning in the wild we must extract messy data
stored in arbitrary formats, and preprocess it to suit our needs.  Fortunately,
Perl as a language can do much of the heavy lifting.

This section, while no substitute for knowing proper Perl will give you a crash
course on some of the most common routines. Along with Perl, `PDL` itself has
some user-friendly routines to do file operations that can be very useful.

These routines are part of the [`PDL::IO`](https://metacpan.org/pod/PDL::IO)
high level package that comes pre-installed with PDL.

## Reading the Dataset

Comma-separated values (CSV) files are ubiquitous for the storing of tabular
(spreadsheet-like) data.  In them, each line corresponds to one record and
consists of several (comma-separated) fields, e.g., "Albert Einstein,March 14
1879,Ulm,Federal polytechnic school,field of gravitational physics".  To
demonstrate how to load CSV files with `PDL::IO::*`, we **create a CSV file
below** `./pdl_data/house_tiny.csv`.  This file represents a dataset of homes,
where each row corresponds to a distinct home and the columns correspond to the
number of rooms (`NumRooms`), the roof type (`RoofType`), and the price
(`Price`).

We use the [`Text::CSV_XS`](https://metacpan.org/pod/Text::CSV_XS) module to
read and write a CSV file correctly, especially if your columns have arbitrary
text in them. This module was installed as part of the
[Installation](../chapter_installation/index.md) list of pre-requisites.

```perl
pdl> use Cwd qw(getcwd abs_path)
pdl> use Text::CSV_XS qw(csv)
pdl> mkdir './pdl_data',0755
pdl> print abs_path './pdl_data'
/home/myuser/pdl_data
pdl> $content = [[qw(NumRooms RoofType Price)], [undef, undef, 127500], [2,
undef, 106000], [4, 'Slate', 178100], [undef, undef, 140000]]
pdl> csv(in => $content, out => "./pdl_data/house_tiny.csv", encoding => "UTF-8") 
```
Let's check if the file was written correctly. In a regular shell like `bash` do
the following:

```bash
$ cat ./pdl_data/house_tiny.csv
NumRooms,RoofType,Price 
,,127500 
2,,106000 
4,Slate,178100 
,,140000
```

Now let's load the dataset with `csv` again.

```perl
pdl> use Text::CSV_XS qw(csv)
pdl> use Data::Dumper
pdl> $incontent = csv(in => "./pdl_data/house_tiny.csv", blank_is_undef => 1,
empty_is_undef => 1)
pdl> print Dumper($incontent)
$VAR1 = [
          [
            'NumRooms',
            'RoofType',
            'Price'
          ],
          [
            undef,
            undef,
            '127500'
          ],
          [
            '2',
            undef,
            '106000'
          ],
          [
            '4',
            'Slate',
            '178100'
          ],
          [
            undef,
            undef,
            '140000'
          ]
        ];
```

If we want to directly read each numerical column into `PDL` objects, and the
string columns into Perl arrays, we can use `rcols` from `PDL::IO::Misc` as
shown below, where we ignore the header line, hence setting `LINES => '1:'` and
set the Perl array column to be column `1` which is the `RoofType` column.

```perl
pdl> ($nrooms,$roof,$price) = rcols './pdl_data/house_tiny.csv', { LINES => '1:', COLSEP => ',', PERLCOLS=>[1], TYPES => [ushort,double] }
Reading data into ndarrays of type: [ Ushort Double ]
Read in  4  elements.
pdl> print $nrooms
[0 2 4 0]
pdl> print Dumper($roof)
$VAR1 = [
          undef,
          undef,
          'Slate',
          undef
        ];
pdl> print $price
[127500 106000 178100 140000]
```

Another option is to use `rcsv1D` from `PDL::IO::CSV`. This function **cannot**
read string columns. However, it converts the empty or *missing* values into
`BAD` values in `PDL` which are similar to `NaN`, unlike the `rcols` function
which converts anything that is not a number to `0`.

```perl
pdl> ($nrooms,$price) = rcsv1D './pdl_data/house_tiny.csv', [0,2], {type => [short,double], empty2bad => 1, header => 1}
pdl> print $nrooms
[BAD 2 4 BAD]
pdl> print $price
[127500 106000 178100 140000]
```

Now that we have seen how to use the `rcols` and `rcsv1D` functions to load
columns into PDL objects or _piddles_, we will do the following to convert the
data into a [`Data::Frame`](https://metacpan.org/pod/Data::Frame) object.

To convert PDL objects into a `Data::Frame` object we can do the following:

```perl
pdl> ($nrooms,$roof,$price) = rcols './pdl_data/house_tiny.csv', { LINES => '1:', COLSEP => ',', PERLCOLS=>[1], TYPES => [ushort,double] }
pdl> $nrooms = $nrooms->setbadif($nrooms == 0)
pdl> print $nrooms
[BAD 2 4 BAD]
pdl> use Data::Frame
pdl> $df = Data::Frame->new(columns => [ 'NumRooms' => $nrooms, RoofType => $roof, Price => $price ])
pdl> print $df
-------------------------------
    NumRooms  RoofType  Price  
-------------------------------
 0  BAD                 127500 
 1  2                   106000 
 2  4         Slate     178100 
 3  BAD                 140000 
-------------------------------
```

`Data::Frame` comes with it's own CSV file reader that does **all** of the above
in a **single** call:

```perl
pdl> use Data::Frame
pdl> $df = Data::Frame->from_csv('./pdl_data/house_tiny.csv')
pdl> print $df
-------------------------------
    NumRooms  RoofType  Price  
-------------------------------
 0  BAD                 127500 
 1  2                   106000 
 2  4         Slate     178100 
 3  BAD                 140000 
-------------------------------
```

## Data Preparation

In supervised learning, we train models to predict a designated *target* value,
given some set of *input* values.  Our first step in processing the dataset is
to separate out columns corresponding to input versus target values.  We can
select columns either by name or via integer-location based indexing using the
`at` function.

You might have noticed that `Data::Frame` automatically converts empty or blank
values to `BAD` if the column is numeric, and leaves it as empty or blank if the
column is of string type. `Data::Frame` creates a
[`PDL::SV`](https://metacpan.org/pod/PDL::SV) object for the string columns and
a regular `PDL` object for the numerical ones.

`BAD` values are described
[here](https://metacpan.org/dist/PDL/view/lib/PDL/BadValues.pod) so we will not
be describing it here, but `PDL` allows invalid values to be set to `BAD` which
is different from `NaN` (not a number) which could be a valid value in the
dataset.

*Missing values* or invalid values are important in data science, and must be
handled correctly. Depending upon the context, missing values might be handled
either via *imputation* or *deletion*.  Imputation replaces missing values with
estimates of their values while deletion simply discards either those rows or
those columns that contain missing values. 

Here are some common imputation heuristics. For categorical input fields, a
blank or empty value can be treated as an _unknown_ category.  Since the
`RoofType` column takes values `Slate` and blank, `Data::Frame` can convert this
column into two columns `RoofTypeIsSlate` and `RoofTypeIsUnknown`.  A row whose
roof type is `Slate` will set values of `RoofType_Slate` and `RoofType_unknown`
to 1 and 0, respectively.  The converse holds for a row with a missing
`RoofType` value.

```perl
pdl> $roof = $df->at("RoofType")->unpdl
pdl> print Dumper($roof)
$VAR1 = [
          '',
          '',
          'Slate',
          ''
        ];
pdl> $roof_is_slate = pdl(map { $_ eq 'Slate' ? 1 : 0 } @$roof)
pdl> print $roof_is_slate
[0 0 1 0]

### we can add this as a column now
pdl> $df->add_column("RoofTypeIsSlate", $roof_is_slate)
pdl> print $df
--------------------------------------------
    NumRooms  RoofType  Price   RoofTypeIsSlate 
--------------------------------------------
 0  BAD                 127500  0           
 1  2                   106000  0           
 2  4         Slate     178100  1           
 3  BAD                 140000  0           
--------------------------------------------

## we can also add a column to track unknown roof types
pdl> $df->add_column("RoofTypeIsUnknown", pdl(map { $_ eq '' ? 1 : 0 } @$roof))
pdl> print $df
-----------------------------------------------------------
    NumRooms  RoofType  Price   RoofTypeIsSlate  RoofTypeIsUnknown 
-----------------------------------------------------------
 0  BAD                 127500  0            1             
 1  2                   106000  0            1             
 2  4         Slate     178100  1            0             
 3  BAD                 140000  0            1             
-----------------------------------------------------------
```

For missing numerical values, one common heuristic is to replace the `BAD` or
missing entries with the mean value of the corresponding column.

```perl
pdl> print $df->at("NumRooms")->avg
3
pdl> $df->at("NumRooms")->inplace->setbadtoval(3)
pdl> print $df
-----------------------------------------------------------
    NumRooms  RoofType  Price   RoofTypeIsSlate  RoofTypeIsUnknown 
-----------------------------------------------------------
 0  3                   127500  0            1             
 1  2                   106000  0            1             
 2  4         Slate     178100  1            0             
 3  3                   140000  0            1             
-----------------------------------------------------------
```

Let's now remove the `RoofType` column for making the calculations easier.

```perl
pdl> $df->delete("RoofType")
pdl> print $df
-------------------------------------------------
    NumRooms  Price   RoofIsSlate  RoofIsUnknown 
-------------------------------------------------
 0  3         127500  0            1             
 1  2         106000  0            1             
 2  4         178100  1            0             
 3  3         140000  0            1             
-------------------------------------------------
```

## Conversion to the Tensor Format

Now that [**all the entries in `inputs` and `targets` are numerical,
we can load them into a tensor**] (recall :numref:`sec_ndarray`).

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## Discussion

You now know how to partition data columns, 
impute missing variables, 
and load `pandas` data into tensors. 
In :numref:`sec_kaggle_house`, you will
pick up some more data processing skills. 
While this crash course kept things simple,
data processing can get hairy.
For example, rather than arriving in a single CSV file,
our dataset might be spread across multiple files
extracted from a relational database.
For instance, in an e-commerce application,
customer addresses might live in one table
and purchase data in another.
Moreover, practitioners face myriad data types
beyond categorical and numeric, for example,
text strings, images,
audio data, and point clouds. 
Oftentimes, advanced tools and efficient algorithms 
are required in order to prevent data processing from becoming
the biggest bottleneck in the machine learning pipeline. 
These problems will arise when we get to 
computer vision and natural language processing. 
Finally, we must pay attention to data quality.
Real-world datasets are often plagued 
by outliers, faulty measurements from sensors, and recording errors, 
which must be addressed before 
feeding the data into any model. 
Data visualization tools such as [seaborn](https://seaborn.pydata.org/), 
[Bokeh](https://docs.bokeh.org/), or [matplotlib](https://matplotlib.org/)
can help you to manually inspect the data 
and develop intuitions about 
the type of problems you may need to address.


## Exercises

1. Try loading datasets, e.g., Abalone from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets) and inspect their properties. What fraction of them has missing values? What fraction of the variables is numerical, categorical, or text?
1. Try indexing and selecting data columns by name rather than by column number. The pandas documentation on [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) has further details on how to do this.
1. How large a dataset do you think you could load this way? What might be the limitations? Hint: consider the time to read the data, representation, processing, and memory footprint. Try this out on your laptop. What happens if you try it out on a server? 
1. How would you deal with data that has a very large number of categories? What if the category labels are all unique? Should you include the latter?
1. What alternatives to pandas can you think of? How about [loading NumPy tensors from a file](https://numpy.org/doc/stable/reference/generated/numpy.load.html)? Check out [Pillow](https://python-pillow.org/), the Python Imaging Library. 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17967)
:end_tab:
