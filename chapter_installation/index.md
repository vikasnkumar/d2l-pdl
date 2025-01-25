# Installation

Since we are going to be running using Perl and [PDL](https://pdl.perl.org), we
will have to install all the Perl related modules first. The assumption is that
the reader will be practicing this on Linux, but they are open to experimenting
with Windows or MacOS or other BSD flavors of Unix. However, installing on those
systems is left as an exercise for the reader.


In order to get up and running, we will need an environment for running Python,
the Jupyter Notebook, the relevant libraries, and the code needed to run the
book itself.

## Installation on Linux

For Debian-based Linux distributions, installing `PDL` and Perl is a breeze.

We install the system Perl, but then use `cpanminus` to install the latest
versions of `PDL`. This install can take some time to complete, about 30-45
minutes on a modern system.

We will install all the `PDL` modules in the user's home directory.

```bash
## install system perl
$ sudo apt -y install perl cpanminus \
    liblocal-lib-perl build-essential \
    cmake pkg-config autotools-dev automake \
    autoconf make g++ gfortran swig \
    gnuplot graphviz libjson-xs-perl libdatetime-perl

## install default modules for that perl version
$ PERL_MODULES_VER=$(apt-cache search ^perl-modules | cut -d' ' -f 1)
$ sudo apt -y install $PERL_MODULES_VER
$ which cpanm /usr/bin/cpanm

## create local directory where installation will happen
$ mkdir -p ~/perl5/lib/perl5
$ eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)
$ export PATH=$PATH:$PERL_LOCAL_LIB_ROOT/bin
```

Add the following two lines to your `~/.bashrc` or `~/.zshrc` or `~/.profile`

```text
eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)
export PATH=$PATH:$PERL_LOCAL_LIB_ROOT/bin
```

Now that you have added the two lines to your `~/.bashrc` or similar shell
config file, restart your shell or `source` your `~/.bashrc`

Let's install the remaining packages we need to move forward with using `PDL`
for deep learning.

```bash
$ source ~/.bashrc
$ cpanm PDL PDL::Perldl2 Text::CSV_XS PDL::CCS GraphViz Hash::Ordered \
    Function::Parameters Mouse
## for building AI::MXNet
$ cpanm Alien::SWIG4 File::Which
$ which perldl
/home/myuser/perl5/bin/perldl
$ which pdl2
/home/myuser/perl5/bin/pdl2
```

## Installation on Windows

[Strawberry Perl](https://strawberryperl.com/) is the supported version of Perl
on Windows that supports PDL out of the box. You must install the zip release
that comes with `PDL`. As of this writing it is
[ 5.40.0.1 PDL zip ](https://github.com/StrawberryPerl/Perl-Dist-Strawberry/releases/download/SP_54001_64bit_UCRT/strawberry-perl-5.40.0.1-64bit-PDL.zip)

The rest of the packages can be installed using the Perl or CPAN shell that
comes with the installer, and the same packages as above will be needed.

```cmd
C:\StrawberryPerl\> cpanm PDL::Perldl2 Text::CSV_XS PDL::CCS Hash::Ordered \
    Function::Parameters Mouse
```

## Installing the Deep Learning Framework `mxnet`

Before installing any deep learning framework, please first check whether or not
you have proper GPUs on your machine (the GPUs that power the display on a
standard laptop are not relevant for our purposes).  For example, if your
computer has NVIDIA GPUs and has installed
[CUDA](https://developer.nvidia.com/cuda-downloads), then you are all set.  If
your machine does not house any GPU, there is no need to worry just yet.  Your
CPU provides more than enough horsepower to get you through the first few
chapters.  Just remember that you will want to access GPUs before running larger
models.

**NOTE**: As of 2024, MXNet was retired by the Apache foundation. The last
supported CUDA version in the original MXNet was 11.8. We use a version that we
have modified for CUDA 12.6 and it can be found
[here](https://github.com/selectiveintellect/modified-mxnet). The
[INSTALL.md](https://github.com/selectiveintellect/modified-mxnet/blob/master/INSTALL.md)
shows how to build both the CPU and GPU versions of the C++ library.

The reader must refer to those instructions first and install the `libmxnet.so`
library in a folder that is accessible to the reader, either in `/usr/local` or
in `$HOME/mxnet/` and we will refer to this by the `MXNET_LIB` environment
variable.

To install a GPU-enabled version of MXNet, we need to find out what version of
CUDA you have installed.  You can check this by running `nvcc --version`.
Assume that you have installed CUDA 12.6, the instructions above should be
sufficient.

```bash
$ export MXNET_DIR=$HOME/mxnet/
$ export MXNET_LIB=$HOME/mxnet/lib
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIB
```

## Installing `AI::MXNet`

The Perl package [`AI::MXNet`](https://metacpan.org/pod/AI::MXNet) is a
high-level package to use the MXNet C++ library in Perl with `PDL`. We will
cover this in the upcoming chapters.

After you have setup `PDL` correctly, you need to run the following to install
`AI::MXNet`.

We will install this from our Github
[fork](https://github.com/selectiveintellect/modified-mxnet.git). If you have
followed instructions from the above section, you already have this installed
and have compiled the `libmxnet.so` library successfully, either in CPU mode or
in GPU mode.

The original MXNet code has been archived, so we have been making our own
modifications to keep it working and up-to-date as best as possible.

Remember that the environment variable `MXNET_DIR` has to be set as above to
point to the **installed** version of the built library. This folder will have
the include files and the library files that are needed by the Perl packages.

```bash
$ cd modified-mxnet/perl-package
$ cd AI-NNVMCAPI
$ perl Makefile.PL
$ make
$ make test
$ make install
$ cd ..
```
To test that the installation worked, you can run `perldoc AI::NNVMCAPI`.

Next we install `AI-MXNetCAPI`.

```bash
$ cd modified-mxnet/perl-package
$ cd AI-MXNetCAPI
$ perl Makefile.PL
$ make
$ make test
$ make install
$ cd ..
```
To test that the installation worked, you can run `perldoc AI::MXNetCAPI`.

Next we install `AI-MXNet`

```bash
$ cd modified-mxnet/perl-package
$ cd AI-MXNet
$ perl Makefile.PL
$ make
$ make test
$ make install
$ cd ..
```

[Notation](chapter_notation/index.md)
