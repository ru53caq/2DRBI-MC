# 2DRBI-MC
MC code for computing the free energy ratio of the 2DRBI model for varying points in p-T space.


Building and installation
-------------------------
### Building and installing ALPSCore

Detailed instructions on [how to build ALPSCore][7] can be fournd in the
project's wiki. 

### Building client codes

The proposed framework does not follow a traditional build model, in that it does
_not_ provide a library for the client codes to link against. This is rooted in
the fact that it provides `main` functions for the five executables mentioned in
the foregoing section. Instead, the root-level
[`CMakeLists.txt`](./CMakeLists.txt) exposes the relevant source files as CMake
variables `${TKSVM_SAMPLE_SRC}` (and analogous for the other executables), the
libraries as `${TKSVM_LIBRARIES}` and the include directories as
`${TKSVM_INCLUDE_DIRS}`.
These may then be used from the `CMakeLists.txt` file of each of the client
codes. It must also add the compile definitions exposed in
`${TKSVM_DEFINITIONS}` and additionally define a variable `TKSVM_SIMINCL`
holding the name of the header of the main ALPSCore simulation class which
should define a type (alias) `sim_base` to that class.

Refer to the build file of one of the client codes (_e.g._
[`gauge/CMakeLists.txt`](./gauge/CMakeLists.txt)) for an example. Third-party
client codes may rather keep this repository as a submodule in their own tree,
in which case the line

    add_subdirectory(.. tksvm)

simply becomes

    add_subdirectory(svm-order-params)

Refer to the READMEs of the client codes for build instructions on them
specifically, e.g. [Building the `gauge` client
code](./gauge#building-and-installation).
